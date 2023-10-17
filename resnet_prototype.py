import sys

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
import torch.utils.data as util_data
import tqdm
from torchvision.models import resnet18

from src import FullSpectroAudioSet
from train_config import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
# torch.set_num_threads(23)
dataset = FullSpectroAudioSet(TRAIN_DATA_SET_PATH)
dataset = util_data.Subset(dataset, range(5000))
data_loader = util_data.DataLoader(dataset, shuffle=False, batch_size=100)

res_net = resnet18(num_classes=527)
res_net.to(device)
projection = torch.nn.Conv2d(1, 3, kernel_size=(1, 1))
model = torch.nn.Sequential(projection, res_net)
model.to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(list(res_net.parameters()) + list(projection.parameters()), lr=1e-6)
if sys.platform not in ["win32", "win64"]:
    torch.compile(model)
total_loss = []
for epoch in range(40):
    epoch_loss = []
    for spec, label in tqdm.tqdm(data_loader):
        spec = spec.to(device)
        label = label.to(device)
        out = model(spec)
        optimizer.zero_grad()
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.detach().cpu().numpy())
    print("Epoch: ", epoch, " Loss: ", np.mean(epoch_loss))
    total_loss.append(np.mean(epoch_loss))

plt.plot(total_loss)
plt.show()

torch.save(res_net.state_dict(), "res_net.pt")
torch.save(projection.state_dict(), "projection.pt")
