import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelDepthwise(nn.Module):
    def __init__(self, in_channels, reduce='mean'):
        super().__init__()
        self.in_channels = in_channels
        self.reduce = reduce  # 'mean' or 'max'

        # sobel kernels (depthwise)
        sobel_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)

        sobel_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)

        # stack for depthwise convolution (applied per channel)
        self.sobel_x = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=in_channels, 
            bias=False
        )
        self.sobel_y = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=in_channels, 
            bias=False
        )

        # initialize weights: sobel filters
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(in_channels, 1, 1, 1))
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(in_channels, 1, 1, 1))

        # freeze: sobel is fixed
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # x shape: [batch_size, num_channel, height, width]
        G_x = self.sobel_x(x)  # horizontal gradients
        G_y = self.sobel_y(x)  # vertical gradients

        # gradient magnitude
        gradient_magnitude = torch.sqrt(G_x ** 2 + G_y ** 2)

        # reduce spatial dimensions (mean or max)
        if self.reduce == 'mean':
            out = F.adaptive_avg_pool2d(gradient_magnitude, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.reduce == 'max':
            out = F.adaptive_max_pool2d(gradient_magnitude, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            raise ValueError("reduce must be 'mean' or 'max'")

        return out  # shape: [batch_size, num_channel]