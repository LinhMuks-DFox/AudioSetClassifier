import logging
import os.path
import sys

import torch

import hyper_para
import train_config
import train_prepare

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(train_config.DUMP_PATH, 'train.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.info

model = train_prepare.make_model()
dataset = train_prepare.make_dataset()
train_loader, validate_loader, test_loader = train_prepare.make_dataloader(dataset)
device = torch.device(hyper_para.DEVICE)
loss_function = train_prepare.make_loss_function()
optimizer = train_prepare.make_optimizer(model)
scheduler = train_prepare.make_scheduler(optimizer)


def train():
    for epoch in range(hyper_para.EPOCH):
        for i, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = one_step_loss(data, label)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                log(f'epoch: {epoch}, step: {i}, loss: {loss.item()}')
        scheduler.step()


def one_step_loss(data: torch.Tensor, label: torch.Tensor):
    data = data.to(device)
    label = label.to(device)
    output = model(data)
    loss = loss_function(label, output)
    return loss


def validate():
    pass


def test():
    pass


def main():
    pass
