import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model_openai import my_load
from zero_shot.zero_shot_classifier import ZeroShotClassifier
from model_openai import SimpleTokenizer
from zero_shot.zero_shot_metadata_imagenet import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm
import random
from typing import Union, List
from types import MethodType


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            projection_dims: int = 1024,
            temperature: float = 1.0,
            weight_decay: float = 0.0,
            lr_visual: float = 1e-3,
            lr_text: float = 5e-4,
            lr_warmup_epochs: int = 5,
            batch_size: int = 64,
            old_checkpoint_path: Union[str, List[str]] = None,
            current_task: int = 0,
            batch_size_zs: int = 256,
            zero_shot_eval_interval: int = 5,
            recall_eval_interval: int = 5,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def setup_s4pool(self):
        def conv_s4pool(self, input):
            (shift1, shift2) = (np.random.randint(2), np.random.randint(2))
            if input.shape[2] % 2 == 1:
                padded_input = F.pad(input, (0, 0, self.padding[0], self.padding[0]), "constant", 0)
                shift1 = 0
            else:
                first_side = self.padding[0] - shift1
                second_side = self.padding[0] + shift1 - 1
                padded_input = F.pad(input, (0, 0, first_side, second_side), "constant", 0)
            if input.shape[3] % 2 == 1:
                shift2 = 0
                padded_input = F.pad(padded_input, (self.padding[1], self.padding[1], 0, 0), "constant", 0)
            else:
                first_side = self.padding[1] - shift2
                second_side = self.padding[1] + shift2 - 1
                padded_input = F.pad(padded_input, (first_side, second_side, 0, 0), "constant", 0)
            fmap = F.conv2d(padded_input,
                            weight=self.weight.to(self.device),
                            bias=self.bias,
                            stride=self.stride,
                            dilation=self.dilation,
                            groups=self.groups)
            return fmap

        def s4_maxpool(self, input):
            (shift1, shift2) = (np.random.randint(2), np.random.randint(2))
            if input.shape[2] % 2 == 1:
                padded_input = F.pad(input, (0, 0, self.padding, self.padding), "constant", float('-inf'))
            else:
                first_side = self.padding - shift1
                second_side = self.padding + shift1 - 1
                padded_input = F.pad(input, (0, 0, first_side, second_side), "constant", float('-inf'))
            if input.shape[3] % 2 == 1:
                padded_input = F.pad(padded_input, (self.padding, self.padding, 0, 0), "constant", float('-inf'))
            else:
                first_side = self.padding - shift2
                second_side = self.padding + shift2 - 1
                padded_input = F.pad(padded_input, (first_side, second_side, 0, 0), "constant", float('-inf'))
            return F.max_pool2d(padded_input, self.kernel_size, self.stride,
                                0, self.dilation, self.ceil_mode,
                                self.return_indices)

        def conv_s4pool_1x1(self, input):
            (shift1, shift2) = (np.random.randint(2), np.random.randint(2))
            fmap = F.conv2d(input[:, :, shift1:, shift2:],
                            weight=self.weight.to(self.device),
                            bias=self.bias,
                            stride=self.stride,
                            dilation=self.dilation,
                            groups=self.groups)
            return fmap


        def modify_conv_module(mod):
            if isinstance(mod, torch.nn.Conv2d) and mod.kernel_size[0] > 1 and mod.stride == (2, 2):
                mod.forward = MethodType(conv_s4pool, mod)
            if isinstance(mod, torch.nn.Conv2d) and mod.kernel_size[0] == 1 and mod.stride == (2, 2):
                mod.forward = MethodType(conv_s4pool_1x1, mod)
            if isinstance(mod, torch.nn.MaxPool2d):
                print("changed maxpool")
                mod.forward = MethodType(s4_maxpool, mod)

        self.model.apply(modify_conv_module)





