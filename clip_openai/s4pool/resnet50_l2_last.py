import os
import torchmetrics
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from args import parse_args
import pytorch_lightning as pl
from imagenet_dali import ClassificationDALIDataModule
from torchvision.models import resnet50


def unified_net():
    u_net = resnet50(pretrained=False)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    return u_net




class ResNet50_L2(LightningModule):
    def __init__(self):
        super().__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), resnet50(pretrained=False).layer1
        )
        self.unified_net = unified_net()
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)
        self.unified_size = (56, 56)

    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)

        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')

        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)

        return z1, z2, z3, y1, y2, y3

    def forward_32(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        z1 = self.small_net(small_imgs)
        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        y1 = self.unified_net(z1)

        return y1

    def forward_128(self, imgs):
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        z2 = self.mid_net(mid_imgs)
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')
        y2 = self.unified_net(z2)

        return y2

    def forward_224(self, imgs):
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')
        z3 = self.large_net(large_imgs)
        y3 = self.unified_net(z3)

        return y3


class ResNet50(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = ResNet50_L2()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.metrics_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def share_step(self, batch, batch_idx):
        x, y = batch
        z1, z2, z3, y_hat1, y_hat2, y_hat3 = self(x)

        ce_loss1 = self.ce_loss(y_hat1, y)
        ce_loss2 = self.ce_loss(y_hat2, y)
        ce_loss3 = self.ce_loss(y_hat3, y)

        si_loss1 = self.mse_loss(z1, z2)
        si_loss2 = self.mse_loss(z1, z3)
        si_loss3 = self.mse_loss(z2, z3)

        if si_loss1 < 0.01:
            si_loss1 = 0

        if si_loss2 < 0.01:
            si_loss2 = 0

        if si_loss3 < 0.01:
            si_loss3 = 0

        total_loss = si_loss1 + si_loss2 + si_loss3 + ce_loss1 + ce_loss2 + ce_loss3

        acc1 = self.metrics_acc(y_hat1, y)
        acc2 = self.metrics_acc(y_hat2, y)
        acc3 = self.metrics_acc(y_hat3, y)

        result_dict = {
            "si_loss1": si_loss1,
            "si_loss2": si_loss2,
            "si_loss3": si_loss3,
            "ce_loss1": ce_loss1,
            "ce_loss2": ce_loss2,
            "ce_loss3": ce_loss3,
            "total_loss": total_loss,
            "acc1": acc1,
            "acc2": acc2,
            "acc3": acc3,
        }
        return result_dict

    def training_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        train_result_dict = {f'train_{k}': v for k, v in result_dict.items()}
        self.log_dict(train_result_dict, on_epoch=True)
        return result_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        val_result_dict = {f'val_{k}': v for k, v in result_dict.items()}
        self.log_dict(val_result_dict, on_epoch=True)
        return val_result_dict




class ModifiedResNet(LightningModule):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x