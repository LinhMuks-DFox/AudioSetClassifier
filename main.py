import logging
import os.path
import shutil
import sys

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm

import hyper_para
import src.tags
import train_config
import train_prepare
from src.ClassifierTester import ClassifierTester


# region logger config
def compose_path(file):
    return os.path.join(train_config.DUMP_PATH, file)


if not os.path.exists(train_config.DUMP_PATH):
    print("DUMP PATH NOT EXISTS, CREATING...")
    os.makedirs(train_config.DUMP_PATH)
else:
    print("DUMP PATH EXISTS, SKIPPING...")

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
        self.model = train_prepare.make_classifier()
        self.dataset = train_prepare.make_dataset()
        if train_config.DRY_RUN:
            self.dataset = torch.utils.data.Subset(self.dataset, range(hyper_para.DRY_RUN_DATE_SET_LENGTH))
        self.train_loader, self.validate_loader, self.test_loader = train_prepare.make_dataloader(self.dataset)
        self.device = train_prepare.select_device()
        self.loss_function = train_prepare.make_loss_function()
        self.optimizer = train_prepare.make_optimizer(self.model)
        self.scheduler = train_prepare.make_scheduler(self.optimizer)
        self.validate_loss, self.train_loss = [torch.empty(0).to(self.device) for _ in range(2)]
        self.classifier_tester: ClassifierTester = ClassifierTester(self.model, self.device, True)

        self.model.to(self.device)
        self.check_point_iota_: int = 0

        self.eval_result = None

    @src.tags.stable_api
    def one_step_loss(self, data: torch.Tensor, label: torch.Tensor):
        data = data.to(self.device)
        label = label.to(self.device)
        output = self.model(data)
        loss = self.loss_function(label, output)
        return loss

    @src.tags.stable_api
    def train(self):
        epoch_cnt = hyper_para.DRY_RUN_EPOCHS if train_config.DRY_RUN else hyper_para.EPOCHS
        for epoch in range(epoch_cnt):
            log(f"train epoch: {epoch} start.")
            epoch_loss = torch.empty(0).to(self.device)
            loss: torch.Tensor

            for x, y in tqdm.tqdm(self.train_loader):
                loss: torch.Tensor = self.one_step_loss(x, y)
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                epoch_loss = torch.hstack((epoch_loss, loss))

            self.epoch_validate()
            self.scheduler.step()
            self.train_loss = torch.hstack((self.train_loss, mean_loss := torch.mean(epoch_loss)))
            log(f"train epoch: {epoch}, mean loss: {mean_loss}")
            log(f"train epoch: {epoch} end.")

    @src.tags.stable_api
    def epoch_validate(self):
        with torch.no_grad():
            _vali_loss = torch.empty(0).to(self.device)
            for i, (data, label) in enumerate(self.validate_loader):
                loss = self.one_step_loss(data, label)
                _vali_loss = torch.hstack((_vali_loss, loss))
            self.validate_loss = torch.hstack((self.validate_loss, torch.mean(_vali_loss)))

    @src.tags.stable_api
    def eval_model(self):
        self.eval_result = (self.classifier_tester
                            .set_dataloader(self.test_loader, hyper_para.CLASS_CNT)
                            .evaluate_model())

    @src.tags.stable_api
    def dump_checkpoint(self):
        torch.save(self.model.state_dict(), compose_path(f"checkpoint{self.check_point_iota_}.pt"))
        self.check_point_iota_ += 1

    @src.tags.stable_api
    def dump_result(self):
        with open(compose_path("train_loss.txt"), "w") as train_f, \
                open(compose_path("validate_loss.txt"), "w") as vali_f, \
                open(compose_path("eval_result.txt"), "w") as eval_f:
            train_f.write("\n".join([f"epoch: {idx}, train loss: {item}"
                                     for idx, item in enumerate(self.train_loss.tolist())]))
            vali_f.write("\n".join([f"epoch: {idx}, validate loss: {item}"
                                    for idx, item in enumerate(self.validate_loss.tolist())]))

            for measure, score in self.eval_result.items():
                eval_f.write(f"{measure}: {score}\n")

        torch.save(self.train_loss, compose_path("train_loss.pt"))
        torch.save(self.validate_loss, compose_path("validate_loss.pt"))

        # plt.matshow(self.eval_result["confusion_matrix"])
        # plt.savefig(compose_path("confusion_matrix.png"), dpi=300)
        # plt.clf()
        log(self.eval_result)
        plt.plot(self.train_loss.detach().cpu().numpy())
        plt.plot(self.validate_loss.detach().cpu().numpy())
        plt.savefig(compose_path("train_validate_loss.png"), dpi=300)
        plt.clf()

    @src.tags.stable_api
    def main(self):
        # region log config
        log(train_config.TRAIN_CONFIG_SUMMARY)
        log(hyper_para.TRAIN_HYPER_PARA_MESSAGE)
        log("Running Dry Run Mode" if train_config.DRY_RUN else "Running Normal Mode")
        log("Dataset length: " + str(len(self.dataset)))
        log("Datashape: " + str(self.dataset[0][0].shape))
        log("Back up train_config.py and hyper_para.py")
        # endregion

        # region backup
        shutil.copy("train_config.py", compose_path("train_config.py.backup"))
        shutil.copy("hyper_para.py", compose_path("hyper_para.py.backup"))
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
            log("Training failed. Error as follows:")
            log(e, exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()
            exit(-1)
        # endregion

        # region eval
        try:
            self.eval_model()
            self.dump_result()

        except Exception as e:
            log("Model evaluation failed. Error as follows:")
            log(e, exc_info=True)
            log(f"Dumping checkpoint... to checkpoint_{self.check_point_iota_}.pt")
            self.dump_checkpoint()
            log(
                f"classifier_tester.multi_label_: {self.classifier_tester.multi_label_}\n"
                f"classifier_tester.confusion_calculate_kernel_: {self.classifier_tester.confusion_calculate_kernel_}\n"
                f"classifier_tester.y_predict_.shape: {self.classifier_tester.y_predict_.shape}\n"
                f"self.classifier_tester.y_true_.shape: {self.classifier_tester.y_true_.shape}\n"
            )

            exit(-1)
        # endregion
        log("Training finished.")


if __name__ == '__main__':
    train_app = TrainApp()
    train_app.main()
