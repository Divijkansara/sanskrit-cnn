"""
The CNN architecture.

SanskritCNN is a compact VGG-style network sized for 32x32 grayscale input and
46 output classes. Three convolutional blocks progressively halve the spatial
resolution (32 -> 16 -> 8 -> 4) while widening the channels (32 -> 64 -> 128),
then a global-average-pooled head produces the class logits.

Design notes
------------
* Two 3x3 convolutions per block give a 5x5 effective receptive field with
  fewer parameters and one extra non-linearity versus a single 5x5 kernel.
* BatchNorm after every convolution keeps activations well-scaled, which
  matters because handwriting stroke thickness varies a lot in this dataset.
* Dropout increases with depth (0.25 -> 0.30 -> 0.35 -> 0.50); the deeper
  layers hold most of the parameters and overfit first.
* The head pools the final 4x4 map down to 2x2 rather than flattening it.
  Flattening 128x4x4 into a dense layer would put ~500K parameters in a single
  matrix and tie the model to exact character placement; pooling to 2x2 keeps
  a coarse sense of *where* a stroke sits (useful, since Devanagari characters
  differ in the position of dots and descenders) at a quarter of the cost.
  Total: ~430K trainable parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg


def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    """[Conv-BN-ReLU] x2 -> MaxPool -> Dropout"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class SanskritCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = cfg.NUM_CLASSES,
        in_channels: int = cfg.NUM_CHANNELS,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            conv_block(in_channels, 32, 0.25),   # 32x32 -> 16x16
            conv_block(32, 64, 0.30),            # 16x16 -> 8x8
            conv_block(64, 128, 0.35),           # 8x8   -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),             # 128 x 4 x 4 -> 128 x 2 x 2
            nn.Flatten(),                        # -> 512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int = cfg.NUM_CLASSES) -> SanskritCNN:
    return SanskritCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = build_model()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy = torch.randn(4, cfg.NUM_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    print(model)
    print(f"\ntrainable parameters: {n:,}")
    print("output shape:", tuple(model(dummy).shape))
