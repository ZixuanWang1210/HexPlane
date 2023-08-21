import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight_dim1=1.0, TVLoss_weight_dim2=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight_dim1 = TVLoss_weight_dim1
        self.TVLoss_weight_dim2 = TVLoss_weight_dim2

    """
    1. **获取张量尺寸**：
        - `batch_size`：批量大小，即输入张量`x`的第一个维度。
        - `h_x` 和 `w_x`：分别是输入张量`x`的高度和宽度（通常对应于图像或特征图）。

    2. **计算有效元素数量**：
        - `count_h`：高度方向上差值的总数量。
        - `count_w`：宽度方向上差值的总数量。

    3. **计算高度和宽度方向上的TV Loss**：
        - `h_tv`：计算输入张量`x`沿高度方向（第3个维度）的总变分损失。具体地，它计算相邻像素间的差的平方和，然后乘以一个权重`self.TVLoss_weight_dim1`。
        - `w_tv`：同样地，计算输入张量`x`沿宽度方向（第4个维度）的总变分损失。同样使用一个权重`self.TVLoss_weight_dim2`。

    4. **平均并返回结果**：
        - 先对`h_tv`和`w_tv`进行归一化，除以它们各自的有效元素数量（`count_h`和`count_w`）。
        - 然后将两者相加，并除以`batch_size`，以得到批量中每个样本的平均总变分损失。
        - 最后乘以2（可能是为了调整损失的规模，但这取决于具体的应用场景）。
    """
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = (
            torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
            * self.TVLoss_weight_dim1
        )
        w_tv = (
            torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
            * self.TVLoss_weight_dim2
        )
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


@torch.jit.script
def compute_dist_loss(weights, svals):
    """Compute the distortion loss of each ray.
    Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields.
        Barron et al., CVPR 2022.
        https://arxiv.org/abs/2111.12077
    As per Equation (15) in the paper. Note that we slightly modify the loss to
    account for "sampling at infinity" when rendering NeRF.
    Args:
        pred_weights (jnp.ndarray): (..., S) predicted weights of each
            sample along the ray.
        svals (jnp.ndarray): (..., S + 1) normalized marching step of each
            sample along the ray.
    """

    smids = 0.5 * (svals[..., 1:] + svals[..., :-1])
    sdeltas = svals[..., 1:] - svals[..., :-1]

    loss_uni = (1 / 3) * (sdeltas * weights.pow(2)).sum(dim=-1).mean()
    wm = weights * smids
    w_cumsum = weights.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = weights[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni
