import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze()