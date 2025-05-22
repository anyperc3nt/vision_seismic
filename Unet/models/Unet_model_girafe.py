import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, transpose=False
    ):
        super().__init__()

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.norm.weight, 1)
        # nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=2),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.blocks = nn.ModuleList(
            UNetEncoderBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(
                hidden_channels[:-1],
                hidden_channels[1:],
            )
        )

    def forward(self, x):
        outputs = [x]

        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        return outputs


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = ConvBlock(
            in_channels,
            out_channels,
            stride=2,
            transpose=True,
        )
        self.convs = nn.Sequential(
            ConvBlock(2 * out_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # Приводим размеры skip-соединения к размерам x (upsampled)
        skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        return self.convs(x)


class UNetDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.blocks = nn.ModuleList(
            UNetDecoderBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(
                hidden_channels[:-1],
                hidden_channels[1:],
            )
        )

    def forward(self, encoder_outputs):
        x, *skip_connections = reversed(encoder_outputs)
        for block, skip_connection in zip(self.blocks, skip_connections):
            x = block(x, skip_connection)

        return x


class UNet_girafe(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=None,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [8, 32, 64, 128, 256]

        out_channels = hidden_channels[0]
        self.input_convs = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

        self.encoder = UNetEncoder(hidden_channels)
        self.decoder = UNetDecoder(hidden_channels[::-1])

        in_channels = hidden_channels[0]
        self.to_mask = nn.Conv2d(in_channels, out_channels=num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.input_convs(x)
        encoder_outputs = self.encoder(x)
        x = self.decoder(encoder_outputs)

        x = self.to_mask(x)
        return self.activation(x)
