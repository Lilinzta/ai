from torch import nn


class NET(nn.Module):
    def __init__(self, num_channels=1):
        super(NET, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2),
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.net(x)
        return x
        # return x.argmax(dim = 1)
