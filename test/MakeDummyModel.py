import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm


class DummyModelForMNIST(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196, 98),
            nn.ReLU(),
            nn.Linear(98, 49),
            nn.ReLU(),
            nn.Linear(49, 10)
        )

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        self.dataset = datasets.MNIST(root='data', train=True, transform=self.transform, download=True)
        self.train_set, self.test_set, self.validate_set = (
            torch.utils.data.random_split(self.dataset, [0.6, 0.2, 0.2]))
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64, shuffle=True)
        self.validate_loader = torch.utils.data.DataLoader(self.validate_set, batch_size=64, shuffle=True)

        self.model = DummyModelForMNIST()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def main(self):
        self.model.to(self.device)
        for epoch in tqdm.tqdm(range(10)):
            print(f"epoch {epoch} start")
            _epoch_loss = torch.empty(0).to(self.device)
            for x, y in tqdm.tqdm(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                y_predict = self.model(x)
                loss = self.loss(y_predict, y)
                loss.backward()
                self.optimizer.step()
                _epoch_loss = torch.hstack((_epoch_loss, loss))
            print(f"epoch {epoch} end, loss: {torch.mean(_epoch_loss).item()}")

        torch.save(self.model.state_dict(), "dummy_model.pt")


if __name__ == '__main__':
    Trainer().main()
