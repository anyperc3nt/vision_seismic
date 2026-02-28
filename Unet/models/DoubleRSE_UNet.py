import torch
import torch.nn as nn

"""
Unet с макс пулингом и R2SE блоками
"""


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # Гарантируем, что промежуточный слой будет хотя бы 1 канал
        reduced_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Масштабирование каналов
    
class R2SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        
        # Первый residual блок
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(4, channels)
        
        # Второй residual блок
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(4, channels)
        
        # SE блок
        self.se = SEBlock(channels, reduction)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        out = self.se(out)
        
        out += residual
        
        return self.relu(out)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groupnorm_n,
        kernel_size=3,
        stride=1,
        padding=1,
        output_padding=1,
        transpose=False,
    ):
        super().__init__()

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # self.norm = nn.BatchNorm2d(out_channels)
        # self.norm = nn.InstanceNorm2d(out_channels)
        self.norm = nn.GroupNorm(groupnorm_n, out_channels)  # лучше чем InstanceNorm2d
        # self.act = nn.ReLU(inplace=True)
        self.r2se = R2SEBlock(out_channels)  

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        #x = self.act(x)
        x = self.r2se(x)
        return x


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_n):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, groupnorm_n),
            ConvBlock(out_channels, out_channels, groupnorm_n),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, hidden_channels, groupnorm_nums):
        super().__init__()
        self.blocks = nn.ModuleList(
            UNetEncoderBlock(in_channels, out_channels, groupnorm_n)
            for in_channels, out_channels, groupnorm_n in zip(hidden_channels[:-1], hidden_channels[1:], groupnorm_nums)
        )

    def forward(self, x):
        outputs = [x]

        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        return outputs


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_n):
        super().__init__()
        self.upsample = ConvBlock(in_channels, out_channels, groupnorm_n, stride=2, transpose=True)
        self.convs = nn.Sequential(
            ConvBlock(2 * out_channels, out_channels, groupnorm_n),
            ConvBlock(out_channels, out_channels, groupnorm_n),
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.convs(x)


class UNetDecoder(nn.Module):
    def __init__(self, hidden_channels, groupnorm_nums):
        super().__init__()
        self.blocks = nn.ModuleList(
            UNetDecoderBlock(in_channels, out_channels, groupnorm_n)
            for in_channels, out_channels, groupnorm_n in zip(
                hidden_channels[:-1],
                hidden_channels[1:],
                groupnorm_nums
            )
        )

    def forward(self, encoder_outputs):
        x, *skip_connections = reversed(encoder_outputs)
        for block, skip_connection in zip(self.blocks, skip_connections):
            x = block(x, skip_connection)

        return x


class DoubleRSE_UNet(nn.Module):
    def __init__(self, in_channels, num_classes, faults):
        """
        faults: {True, False} сообщает модели о том, что на последний канал не надо накладывать nn.sigmoid()
        """
        super().__init__()

        self.faults = faults
        
        hidden_channels = [8, 16, 32, 64, 128, 256]
        groupnorm_nums = [4, 4, 4, 4, 4]

        out_channels = hidden_channels[0]
        self.input_convs = nn.Sequential(
            ConvBlock(in_channels, out_channels, 8),
            ConvBlock(out_channels, out_channels, 8),
        )

        self.encoder = UNetEncoder(hidden_channels, groupnorm_nums)
        self.decoder = UNetDecoder(hidden_channels[::-1], groupnorm_nums[::-1])

        in_channels = hidden_channels[0]
        self.to_mask = nn.Conv2d(in_channels, out_channels=num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.input_convs(x)
        encoder_outputs = self.encoder(x)
        x = self.decoder(encoder_outputs)
        x = self.to_mask(x)
        if self.faults:
            # Apply sigmoid to all channels except the last
            sig = self.activation(x[:, :-1, :, :])
            last = x[:, -1:, :, :]
            x = torch.cat([sig, last], dim=1)
        else:
            x = self.activation(x)
        return x
