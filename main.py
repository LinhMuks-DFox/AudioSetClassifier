import logging
import os.path
import sys

import torch

import hyper_para
import train_config
import train_prepare
import src.tags

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(train_config.DUMP_PATH, 'train.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.info

model = train_prepare.make_classifier()
dataset = train_prepare.make_dataset()
train_loader, validate_loader, test_loader = train_prepare.make_dataloader(dataset)
device = train_prepare.select_device()
loss_function = train_prepare.make_loss_function()
optimizer = train_prepare.make_optimizer(model)
scheduler = train_prepare.make_scheduler(optimizer)
validata_loss, train_loss, test_loss = [torch.empty(0).to(device) for _ in range(3)]


@src.util.untested
def train():
    for epoch in range(hyper_para.EPOCH):
        log(f"Epoch {epoch} start.")
        _epoch_loss = torch.empty(0)
        for i, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = one_step_loss(data, label)
            loss.backward()
            optimizer.step()
            _epoch_loss = torch.hstack((_epoch_loss, loss))
        train_loss: torch.Tensor = torch.hstack((train_loss, torch.mean(_epoch_loss)))
        scheduler.step()
        log(f"Epoch {epoch} end.")


@src.util.untested
def one_step_loss(data: torch.Tensor, label: torch.Tensor):
    data = data.to(device)
    label = label.to(device)
    output = model(data)
    loss = loss_function(label, output)
    return loss


@src.util.untested
def validate():
    with torch.no_grad():
        _vali_loss = torch.empty(0)
        for i, (data, label) in enumerate(validate_loader):
            loss = one_step_loss(data, label)
            _vali_loss = torch.hstack((_vali_loss, loss))
        validata_loss.append(torch.mean(_vali_loss))


@src.util.untested
def test():
    with torch.no_grad():
        for i, (data, label) in enumerate(validate_loader):
            loss = one_step_loss(data, label)
            test_loss = torch.hstack((test_loss, loss))


@src.util.untested
def main():
    pass
