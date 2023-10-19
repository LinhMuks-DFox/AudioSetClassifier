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
from src.MultiLabelClassifierTester import MultiLabelClassifierTester

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
            self.train_dataset_ = torch.utils.data.Subset(self.train_dataset_,
                                                          range(hyper_para.DRY_RUN_DATE_SET_LENGTH))
            self.validate_test_dataset_ = torch.utils.data.Subset(self.validate_test_dataset_,
                                                                  range(hyper_para.DRY_RUN_DATE_SET_LENGTH))
        self.train_loader_ = train_prepare.make_train_loader(self.train_dataset_)
        self.validate_loader_, self.test_loader_ = train_prepare.make_test_validate_loader(self.validate_test_dataset_)
        self.device_ = train_prepare.select_device()
        self.loss_function_ = train_prepare.make_loss_function()
        self.optimizer_ = train_prepare.make_optimizer(self.model_)
        self.scheduler_ = train_prepare.make_scheduler(self.optimizer_)
        self.validate_loss_, self.train_loss = [torch.empty(0).to(self.device_) for _ in range(2)]
        self.classifier_tester_ = MultiLabelClassifierTester(self.model_,
                                                             self.device_,
                                                             threshold=hyper_para.THRESHOLD,
                                                             use_sigmoid=hyper_para.USE_SIGMOID)
        self.model_.to(self.device_)
        self.check_point_iota_: int = 0

        self.eval_result_ = None
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

    def train(self):
        epoch_cnt = hyper_para.DRY_RUN_EPOCHS if train_config.DRY_RUN else hyper_para.EPOCHS
        for epoch in range(epoch_cnt):
            log(f"train epoch: {epoch} start.")
            self.model_.train()
            epoch_loss = torch.empty(0).to(self.device_)
            loss: torch.Tensor
            log(f"learning rate in this epoch: {self.optimizer_.param_groups[0]['lr']}")
            for x, y in tqdm.tqdm(self.train_loader_):
                loss = self.one_step_loss(x, y)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                epoch_loss = torch.hstack((epoch_loss, loss.detach().clone()))
            self.train_loss = torch.hstack((self.train_loss, mean_loss := torch.mean(epoch_loss)))
            log(f"train epoch: {epoch} end, mean loss: {mean_loss}")
            self.epoch_validate()
            self.scheduler_.step()

    def epoch_validate(self):
        log("epoch validate start.")
        self.model_.eval()
        with torch.no_grad():
            _vali_loss = torch.empty(0).to(self.device_)
            for data, label in tqdm.tqdm(self.validate_loader_):
                loss = self.one_step_loss(data, label)
                _vali_loss = torch.hstack((_vali_loss, loss))
            self.validate_loss_ = torch.hstack((self.validate_loss_, torch.mean(_vali_loss)))
            log(f"validate done, validate loss: {torch.mean(_vali_loss)}")

    def eval_model_dump_eval_result(self):
        self.eval_result_ = (self.classifier_tester_
                             .set_dataloader(self.test_loader_, hyper_para.CLASS_CNT)
                             .evaluate_model())
        confusion_matrix = torch.tensor(self.eval_result_.get("confusion_matrix"))
        torch.save(confusion_matrix, compose_path("confusion_matrix.pt"))
        torch.save(self.classifier_tester_.y_true_, compose_path("tester_y_true.pt"))
        torch.save(self.classifier_tester_.y_predict_, compose_path("tester_y_predict.pt"))
        torch.save(self.classifier_tester_.y_predict_binary_, compose_path("tester_y_predict_binary.pt"))
        with open(compose_path("eval_result.txt"), "w") as f, open(compose_path("confusion_matrix.txt"), "w") as f2:
            f.write(f"accuracy: {self.eval_result_.get('accuracy')}\n")
            f.write(f"precision: {self.eval_result_.get('precision')}\n")
            f.write(f"recall: {self.eval_result_.get('recall')}\n")
            f.write(f"f1_score: {self.eval_result_.get('f1_score')}\n")
            f.write(f"hamming_loss: {self.eval_result_.get('hamming_loss')}\n")
            f.write(self.classifier_tester_.classification_report())
            for i in range(confusion_matrix.shape[0]):
                f2.write("Confusion matrix for class " + self.class2label[f"{i}"]["display_name"] + "\n")
                f2.write("\n".join([str(item) for item in confusion_matrix[i].tolist()]) + "\n")
                f2.write("--------------------\n")

    def dump_checkpoint(self, name: str = None):
        if name is None:
            name = f"checkpoint{self.check_point_iota_}.pt"
            self.check_point_iota_ += 1
        torch.save(self.model_.state_dict(), compose_path(name))

    def dump_result(self):
        with open(compose_path("train_loss.txt"), "w") as train_f, \
                open(compose_path("validate_loss.txt"), "w") as vali_f:
            train_f.write("\n".join([f"epoch: {idx}, train loss: {item}"
                                     for idx, item in enumerate(self.train_loss.tolist())]))
            vali_f.write("\n".join([f"epoch: {idx}, validate loss: {item}"
                                    for idx, item in enumerate(self.validate_loss_.tolist())]))

        torch.save(self.train_loss, compose_path("train_loss.pt"))
        torch.save(self.validate_loss_, compose_path("validate_loss.pt"))
        log(self.eval_result_)
        plt.plot(self.train_loss.detach().cpu().numpy(), label="train loss")
        plt.plot(self.validate_loss_.detach().cpu().numpy(), label="validate loss")
        plt.title("Train and Validate Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(compose_path("train_validate_loss.png"), dpi=300)
        plt.clf()

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
            self.train()
        except Exception as e:
            log("Training failed. Error as follows:\n" + f"{e}", exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()
            exit(-1)
        # endregion

        # region eval and dump checkpoint
        try:
            log("Evaluating model...")
            self.eval_model_dump_eval_result()
            log("Evaluated. dump eval result to eval_result.txt.")
        except Exception as e:
            log("Eval failed. Error as follows:\n" + f"{e}", exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()
        # endregion

        # region dump result
        try:
            self.dump_result()
        except Exception as e:
            log("Dump result failed. Error as follows:\n" + f"{e}", exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()
            exit(-1)
        # endregion
        log("Training finished.")

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
