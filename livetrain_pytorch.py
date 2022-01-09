import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

run = neptune.init(project="kaori/practice", 
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjZiMDA2YS02MDM3LTQxZjQtOTE4YS1jODZkMTJjNGJlMDYifQ==",
                    source_files=['livetrain_pytorch.py', 'requirements.txt'])
run["sys/tags"].add(['run-pytorch', 'me:kaori'])  # organize things

parameters = {
    "lr": 1e-2,
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "model_filename": "basemodel",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

#log params
run["config/hyperparameters"] = parameters


class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


model = BaseModel(
    parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]
).to(parameters["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters["lr"])

#Log model, criterion and optimizer name
run["config/model"] = type(model).__name__
run["config/criterion"] = type(criterion).__name__
run["config/optimizer"] = type(optimizer).__name__


data_dir = "data/CIFAR10"
compressed_ds = "./data/CIFAR10/cifar-10-python.tar.gz"
data_tfms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=parameters["bs"], shuffle=True, num_workers=2
)
validset = datasets.CIFAR10(
    data_dir, train=False, transform=data_tfms["train"], download=True
)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=parameters["bs"], num_workers=2
)

dataset_size = {"train": len(trainset), "val": len(validset)}

#Log dataset details
run["config/dataset/path"] = data_dir
run["config/dataset/transforms"] = data_tfms
run["config/dataset/size"] = dataset_size


for i, (x, y) in enumerate(trainloader, 0):
    x, y = x.to(parameters["device"]), y.to(parameters["device"])
    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    #log loss and acc
    run["training/batch/loss"].log(loss)

    run["training/batch/acc"].log(acc)

    loss.backward()
    optimizer.step()


fname = parameters["model_filename"]

# Saving model architecture to .txt
with open(f"./{fname}_arch.txt", "w") as f:
    f.write(str(model))

# Saving model weights .pth
torch.save(model.state_dict(), f"./{fname}.pth")

#log model weights
run[f"io_files/artifacts/{parameters['model_filename']}_arch"].upload(
    f"./{parameters['model_filename']}_arch.txt"
)
run[f"io_files/artifacts/{parameters['model_filename']}"].upload(
    f"./{parameters['model_filename']}.pth"
)


from neptune.new.types import File
import torch.nn.functional as F

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
dataiter = iter(validloader)
images, labels = dataiter.next()
model.eval()


if torch.cuda.is_available():
    model.to("cpu")

n_samples = 50
img = images[:n_samples]
probs = F.softmax(model(img), dim=1)
probs = probs.data.numpy()


for i, ps in enumerate(probs):
    pred = classes[np.argmax(ps)]
    gt = classes[labels[i]]
    description = "\n".join(
        [
            "class {}: {}%".format(classes[n], round(p * 100, 2))
            for n, p in enumerate(ps)
        ]
    )
    # log prediction
    run["images/predictions"].log(
        File.as_image(img[i].squeeze().permute(2, 1, 0).clip(0, 1)),
        name=f"{i}_{pred}_{gt}",
        description=description,
    )