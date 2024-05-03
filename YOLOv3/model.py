import torch
import torch.nn as nn
from torchsummary import summary

# Config:

# Tuple: (out_channels, kernel_size, stride)
# List: ["B" for block, number of repeat]
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)  # bias=False for bn_act=True.
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()  # store layers.
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
            # 3 boxes per cell for 3 scales (small, medium, large), 5 values per box (x, y, w, h, objectness).
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2],
                     x.shape[3])  # (batch_size, 3, (num_classes + 5), width, height).
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale prediction block.
        route_connections = []  # for each residual block.

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue  # skip residual block.

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:  # last residual block.
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)  # concatenate with previous residual block output.
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()  # store layers.
        in_channels = self.in_channels  # 3 for RGB.

        for module in config:  # iterate over config.
            if isinstance(module, tuple):  # if tuple, create CNNBlock.
                out_channels, kernel_size, stride = module  # unpack tuple.
                layers.append(  # append to layers.
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,  # kernel_size = 3 for default.
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels  # update in_channels.

            elif isinstance(module, list):  # if list, create ResidualBlock.
                num_repeats = module[1]  # unpack list.
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))  # append to layers.

            elif isinstance(module, str):  # if str, create ScalePrediction or Upsample.
                if module == "S":  # if str == "S", create ScalePrediction layer.
                    layers += [  # append to layers.
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2  # update in_channels for next layer (ScalePrediction) and
                    # residual block (ResidualBlock) after ScalePrediction layer.

                elif module == "U":  # if str == "U", create Upsample layer.
                    layers.append(nn.Upsample(scale_factor=2), )  # append to layers.
                    in_channels = in_channels * 3  # update in_channels for next layer (ResidualBlock).

        return layers


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes).to(device)

    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device)

    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32,
                                 num_classes + 5)  # shape for each scale prediction block (2, 3, 13, 13, 25).
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16,
                                 num_classes + 5)  # shape for each scale prediction block (2, 3, 26, 26, 25).
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8,
                                 num_classes + 5)  # shape for each scale prediction block (2, 3, 52, 52, 25).
    summary(model, input_size=(3, IMAGE_SIZE, IMAGE_SIZE))
    print("Success!")
