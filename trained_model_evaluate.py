import train_prepare
import torch
from src import MultiLabelClassifierTester

MODEL_PATH = r'pth_bin/2023-10-08-17-58-50/ideal/checkpoint0.pt'
batch_dir = r"pth_bin/2023-10-08-17-58-50/ideal/"
model = train_prepare.make_classifier()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to("cuda:0")
dataset = train_prepare.make_dataset(dataset_type="ideal")
_, _, test_loader = train_prepare.make_dataloader(dataset)
tester = MultiLabelClassifierTester(model, "cuda:0", 0.5)
tester.set_dataloader(test_loader, 527)
tester.evaluate_model()
print(tester.classification_report())
