import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, channels, in_channels=1):
        super().__init__()
        dilation = 1
        self.ext = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer1 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 3
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 3
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 3
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 3
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer5 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=False),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 3
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=dilation, groups=channels,
                      bias=False, dilation=dilation),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.conv_out = nn.Conv2d(in_channels=channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)

        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, bias=True),
        )
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.ext(x)
        x2 = self.layer1(x1) + x1
        x3 = self.layer2(x2) + x2
        x4 = self.layer3(x3) + x3
        x5 = self.layer4(x4) + x4
        x6 = self.layer5(x5) + x5
        x7 = self.conv_out(x6 + x1)
        out = self.reg_conv(x7)
        att = self.att_conv(x7)
        out = out * att
        out = x - out
        return out


class PPNet(nn.Module):
    def __init__(self, in_channels, block_nums):
        super().__init__()
        self.block_nums = block_nums
        channels = 64
        layers = [Block(channels, in_channels) for _ in range(block_nums)]
        self.blocks = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def forward(self, x):
        x = self.blocks(x)
        return x


class PPNet_recursive(nn.Module):
    def __init__(self, in_channels, block_nums):
        super().__init__()
        self.block_nums = block_nums
        channels = 64
        self.blocks = nn.ModuleList([Block(channels, in_channels) for _ in range(block_nums)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out


class Block3(nn.Module):
    def __init__(self, channels, in_channels=1):
        super().__init__()
        self.ext = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer1 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=True, dilation=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=2, groups=channels,
                      bias=True, dilation=2),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=True, dilation=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, groups=channels,
                      bias=True, dilation=3),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=True, dilation=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=4, groups=channels,
                      bias=True, dilation=4),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=True, dilation=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, groups=channels,
                      bias=True, dilation=3),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.layer5 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels,
                      bias=True, dilation=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            # 2
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=2, groups=channels,
                      bias=True, dilation=2),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.conv_out = nn.Conv2d(in_channels=channels, out_channels=in_channels, kernel_size=3, padding=1, bias=True)

        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=True),
        )
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.ext(x)
        x2 = self.layer1(x1) + x1
        x3 = self.layer2(x2) + x2
        x4 = self.layer3(x3) + x3
        x5 = self.layer4(x4) + x4
        x6 = self.layer5(x5) + x5
        x7 = self.conv_out(x6 + x1)
        out = self.reg_conv(x7)
        att = self.att_conv(x7)
        out = out * att
        out = x - out
        return out


class PPNet3(nn.Module):
    def __init__(self, in_channels, block_nums):
        super().__init__()
        self.block_nums = block_nums
        channels = 64
        layers = [Block3(channels, in_channels) for _ in range(block_nums)]
        self.blocks = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def forward(self, x):
        x = self.blocks(x)
        return x


class my_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='sum')
        # self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, out, y):
        loss = self.criterion(out, y)
        return loss / (y.size(0) * 2)


class my_loss_recursive(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, out, y):
        # recursive loss
        loss = self.criterion(out[0], y)
        for i in range(1, len(out)):
            loss += self.criterion(out[i], y)
        return loss / (y.size(0) * 2 * len(out))

