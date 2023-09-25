import logging
import os.path
import sys

import torch
import torch.utils.data
import json
import hyper_para
import train_config
import train_prepare
import src.tags
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
            self.dataset = torch.utils.data.Subset(self.dataset, range(train_config.DRY_RUN_DATE_SET_LENGTH))
        self.train_loader, self.validate_loader, self.test_loader = train_prepare.make_dataloader(self.dataset)
        self.device = train_prepare.select_device()
        self.loss_function = train_prepare.make_loss_function()
        self.optimizer = train_prepare.make_optimizer(self.model)
        self.scheduler = train_prepare.make_scheduler(self.optimizer)
        self.validata_loss, self.train_loss, self.test_loss = [torch.empty(0).to(self.device) for _ in range(3)]
        self.classifier_tester = ClassifierTester(self.model)

    @src.tags.untested
    def one_step_loss(self, data: torch.Tensor, label: torch.Tensor):
        data = data.to(self.device)
        label = label.to(self.device)
        output = self.model(data)
        loss = self.loss_function(label, output)
        return loss

    @src.tags.untested
    def validate(self):
        with torch.no_grad():
            _vali_loss = torch.empty(0)
            for i, (data, label) in enumerate(self.validate_loader):
                loss = self.one_step_loss(data, label)
                _vali_loss = torch.hstack((_vali_loss, loss))
            self.validata_loss = torch.hstack((self.validata_loss, torch.mean(_vali_loss)))

    @src.tags.untested
    def test(self):
        with torch.no_grad():
            for i, (data, label) in enumerate(self.validate_loader):
                loss = self.one_step_loss(data, label)
                self.test_loss = torch.hstack((self.test_loss, loss))

    @src.tags.untested
    def eval_model(self):
        with open(compose_path("model_eval.txt")) as f:
            model_scores = (self.classifier_tester
                            .set_dataloader(self.test_loader, hyper_para.CLASS_CNT)
                            .calculate_confusion_matrix()
                            .calculate_accuracy()
                            .calculate_precision()
                            .calculate_recall()
                            .calculate_f1_score()
                            .status_map())
            json.dump(model_scores, f)

    @src.tags.unfinished_api
    def main(self):
        log(train_config.TRAIN_CONFIG_SUMMARY)
        log("Running Dry Run Mode" if train_config.DRY_RUN else "Running Normal Mode")
        log("Dataset length: " + str(len(self.dataset)))
        log("Datashape: " + str(self.dataset[0][0].shape))
        log("Start training.")
        # train()
        log("Training finished.")


if __name__ == '__main__':
    train_app = TrainApp()
    train_app.main()
