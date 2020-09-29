import torch.nn as nn
import torch

class DepthNetModel(nn.Module):
    def __init__(self):
        super(DepthNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(90, 45, 7, stride=1),
            nn.ReLU(),
            nn.Conv2d(45, 45, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(45, 28, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(28, 1, 1, stride=1),
        )

    def forward(self, features):
        out = self.layer(features)
        return out


class ColorNetModel(nn.Module):
    def __init__(self):
        super(ColorNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(15, 90, 7, stride=1), # 3N + 3, N = 4
            nn.ReLU(),
            nn.Conv2d(90, 90, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(90, 45, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(45, 3, 1, stride=1),
        )

    def forward(self, features):
        out = self.layer(features)
        return out

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.fc1 = nn.Linear(256, 45)
        self.fc2 = nn.Linear(256, 45)
        self.fc3 = nn.Linear(45, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        batch_size = x.size(0)
        return nn.functional.sigmoid(self.decoder(z).view(batch_size)), mu, logvar