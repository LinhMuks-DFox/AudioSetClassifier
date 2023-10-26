import json
import logging
import os.path
import shutil
import sys

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm

import hyper_para
import train_config
import train_prepare
from train_prepare import compose_path
from src.MonoLabelClassifcationTester import MonoLabelClassificationTester

# region logger config


if not os.path.exists(train_config.DUMP_PATH):
    print("DUMP PATH NOT EXISTS, CREATING...")
    os.makedirs(compose_path())
else:
    print("DUMP PATH EXISTS, SKIPPING...")
if "win" in train_config.PLATFORM:
    torch.set_num_threads(train_config.CPU_N_WORKERS)

logging.basicConfig(
    format='%(asctime)s: \n%(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(compose_path("train.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info


# endregion

class TrainApp:
    def __init__(self):
        log("Train Init")
        self.model_ = train_prepare.make_classifier()
        self.train_dataset_ = train_prepare.make_dataset(json_path=train_config.TRAIN_DATA_SET_JSON,
                                                         audio_sample_path=train_config.TRAIN_DATA_SET_PATH)
        self.validate_test_dataset_ = train_prepare.make_dataset(json_path=train_config.EVAL_DATA_SET_JSON,
                                                                 audio_sample_path=train_config.EVAL_DATE_SET_PATH)
        if train_config.DRY_RUN:
            self.train_dataset_ = torch.utils.data.Subset(
                self.train_dataset_,
                range(hyper_para.DRY_RUN_DATE_SET_LENGTH)
            )
            self.validate_test_dataset_ = torch.utils.data.Subset(
                self.validate_test_dataset_,
                range(hyper_para.DRY_RUN_DATE_SET_LENGTH)
            )
        self.train_loader_ = train_prepare.make_train_loader(
            self.train_dataset_,
            hyper_para.DYR_RUN_BATCH_SIZE if train_config.DRY_RUN else None
        )
        self.validate_loader_, self.test_loader_ = train_prepare.make_test_validate_loader(
            self.validate_test_dataset_,
            hyper_para.DYR_RUN_BATCH_SIZE if train_config.DRY_RUN else None
        )
        self.device_ = train_prepare.select_device()
        self.loss_function_ = train_prepare.make_loss_function()
        self.optimizer_ = train_prepare.make_optimizer(self.model_)
        self.scheduler_ = train_prepare.make_scheduler(self.optimizer_)
        self.validate_loss_, self.train_loss = [torch.empty(0).to(self.device_) for _ in range(2)]
        self.classifier_tester_ = MonoLabelClassificationTester(self.model_, self.device_)
        self.classifier_tester_.set_loss_function(self.loss_function_)
        self.model_.to(self.device_)
        self.check_point_iota_: int = 0
        self.epoch_cnt = hyper_para.DRY_RUN_EPOCHS if train_config.DRY_RUN else hyper_para.EPOCHS
        train_prepare.set_torch_random_seed()
        log("Loading class_label_indices.json")
        with open(train_config.CLASS_LABELS_INDICES, "r") as f:
            self.class2label = json.load(f)
        log("Init-done")

    def one_step_loss(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device_)
        label = label.to(self.device_)
        output = self.model_(data)
        loss = self.loss_function_(output, label)
        return loss

    def one_epoch_train(self):
        self.model_.train()
        epoch_loss = torch.empty(0).to(self.device_)
        for x, y in tqdm.tqdm(self.train_loader_):
            self.optimizer_.zero_grad()
            loss = self.one_step_loss(x, y)
            loss.backward()
            self.optimizer_.step()
            epoch_loss = torch.hstack((epoch_loss, loss.detach().clone()))
        self.train_loss = torch.hstack((self.train_loss, mean_loss := torch.mean(epoch_loss)))
        return mean_loss

    def one_epoch_validate(self):
        self.model_.eval()
        self.classifier_tester_.set_dataloader(self.validate_loader_, n_class=hyper_para.CLASS_CNT)
        self.classifier_tester_.evaluate_model()
        self.validate_loss_ = torch.hstack((self.validate_loss_,
                                            torch.mean(self.classifier_tester_.loss_)))
        return torch.mean(self.classifier_tester_.loss_)

    def one_epoch(self, epoch_iota: int):
        log(f"Epoch: {epoch_iota} start.")
        train_loss = self.one_epoch_train()
        validate_loss = self.one_epoch_validate()
        log(f"Epoch({epoch_iota}) end. train loss: {train_loss}, "
            f"validate loss: {validate_loss}, "
            f"learning rate: {self.optimizer_.param_groups[0]['lr']}")

    def final_test_and_dump_result(self):
        train_loss, validate_loss = [x.detach().cpu().numpy() for x in [self.train_loss, self.validate_loss_]]
        plt.plot(train_loss, label="train_loss")
        plt.plot(validate_loss, label="validate_loss")
        plt.legend()
        plt.xlabel("epoch(int)")
        plt.ylabel("loss(float)")
        plt.savefig(compose_path("train_validate_loss.png"), dpi=300)
        plt.clf()

        e = self.classifier_tester_.set_dataloader(
            self.test_loader_,
            n_class=hyper_para.CLASS_CNT
        ).evaluate_model()
        torch.save(e, compose_path("final_test_eval_result.pt"))
        with open(compose_path("final_test_eval_result.txt"), "w") as f:
            f.write("accuracy: " + str(self.classifier_tester_.accuracy_) + "\n")
            f.write("precision: " + str(self.classifier_tester_.precision_) + "\n")
            f.write("recall: " + str(self.classifier_tester_.recall_) + "\n")
            f.write("f1_score: " + str(self.classifier_tester_.f1_score_) + "\n")
            f.write("hamming_loss: " + str(self.classifier_tester_.hamming_loss_) + "\n")
            f.write("\n".join([str(row) for row in self.classifier_tester_.confusion_matrix_]))
        plt.matshow(self.classifier_tester_.confusion_matrix_)
        plt.title("Final test confusion matrix")
        plt.savefig(compose_path("final_test_confusion_matrix.png"), dpi=300)
        plt.clf()

    def dump_checkpoint(self, name: str = None):
        if name is None:
            name = f"checkpoint{self.check_point_iota_}.pt"
            self.check_point_iota_ += 1
        torch.save(self.model_.state_dict(), compose_path(name))

    def main(self):
        # region log configures
        log(train_config.TRAIN_CONFIG_SUMMARY)
        log(hyper_para.TRAIN_HYPER_PARA_SUMMARY)
        log(
            "Train config summary: \n"
            f"Selected train device(selected by train_prepare.select_device()): {self.device_}\n"
            f"{'Running Dry Run Mode' if train_config.DRY_RUN else 'Running Normal Mode'}\n"
            f"Train Dataset: {str(self.train_dataset_)}\n"
            f"Validate Test Dataset: {str(self.validate_test_dataset_)}\n"
            f"Datashape: {self.train_dataset_[0][0].shape}\n"
            f"Back up train_config.py and hyper_para.py\n"
            f"Random seed: {hyper_para.RANDOM_SEED}\n"
        )
        # endregion

        # region backup
        shutil.copy("train_config.py", compose_path("train_config.py"))
        shutil.copy("hyper_para.py", compose_path("hyper_para.py"))
        if hyper_para.DATA_SET == "encoded":
            log("Using dataset encoded, copying AutoEncoder model to dump path.")
            shutil.copytree("./lib/AutoEncoder", compose_path("AutoEncoder"))
            shutil.copy(train_config.AUTO_ENCODER_MODEL_PATH, compose_path("auto_encoder_model.pt"))
        # endregion

        # region train
        log("Start training.")
        try:
            for epoch_iota in range(self.epoch_cnt):
                self.one_epoch(epoch_iota)
                if epoch_iota % hyper_para.SCHEDULAR_STEP_SIZE == 0 and epoch_iota != 0:
                    self.scheduler_.step()
        except Exception as e:
            log("Training failed. Error as follows:\n" + f"{e}", exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()
            exit(-1)
        # endregion

        # region eval and dump checkpoint
        try:
            log("Evaluating model...")
            self.final_test_and_dump_result()
            log("Evaluated. dump eval result to eval_result.txt.")
        except Exception as e:
            log("Eval failed. Error as follows:\n" + f"{e}", exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()

        try:
            log("Dumping train model's checkpoint...")
            self.dump_checkpoint()
            log("Dump done.")
        except Exception as e:
            log("Dump checkpoint failed. Error as follows:\n" + f"{e}", exc_info=True)
            exit(-1)


if __name__ == '__main__':
    train_app = TrainApp()
    train_app.main()
