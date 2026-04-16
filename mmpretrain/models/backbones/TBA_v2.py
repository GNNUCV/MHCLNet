import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule


class LayerScale(BaseModule):
    """
    LayerScale: 帮助深度模型收敛的神技
    """

    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return x * self.gamma


class CoordinateFusion(BaseModule):
    """
    坐标感知融合模块：
    利用 Coordinate Attention 机制，为 Micro 和 Macro 分支生成带有空间位置信息的权重
    """

    def __init__(self, dim, reduction=16):
        super().__init__()
        # 1. 两个分支的特征提取
        # Micro: 3x3 捕捉细胞核
        self.branch_micro = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # Macro: 分解卷积 7x1 + 1x7 捕捉长条形腺管
        self.branch_macro_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)
        self.branch_macro_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)

        # 2. Coordinate Attention 部分
        mip = max(8, dim // reduction)

        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [N, C, H, 1]
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [N, C, 1, W]

        # 共享的变换层，减少参数
        self.conv1 = nn.Conv2d(dim, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()  # Hardswish 比 ReLU 对微调更友好

        # 分别生成 Micro 和 Macro 的关注权重 (2 * dim)
        self.conv_h = nn.Conv2d(mip, dim * 2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, dim * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # --- 分支计算 ---
        x_micro = self.branch_micro(x)

        # Macro 分支串联计算
        x_macro = self.branch_macro_h(x)
        x_macro = self.branch_macro_v(x_macro)

        # --- Coordinate Attention 生成权重 ---
        # 1. 沿方向池化
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        # x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]

        # 2. 拼接处理 (利用空间相关性)
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. 拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, mip, 1, W]

        # 4. 生成权重 (注意这里输出是 2*dim)
        a_h = self.conv_h(x_h).sigmoid()  # [N, 2C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [N, 2C, 1, W]

        # 拆分为 Micro 权重和 Macro 权重
        a_h_micro, a_h_macro = torch.split(a_h, c, dim=1)
        a_w_micro, a_w_macro = torch.split(a_w, c, dim=1)

        # 5. 计算最终的空间权重
        # weights_micro: [N, C, H, W] - 针对每个位置，决定是否关注 Micro 特征
        w_micro = a_h_micro * a_w_micro
        w_macro = a_h_macro * a_w_macro

        # --- 加权融合 ---
        out = (x_micro * w_micro) + (x_macro * w_macro)

        return out + identity  # 加上残差，防止梯度消失


class TBA_Adapter_v2(BaseModule):
    """
    Tri-Branch Anisotropic Adapter v2 (Coordinate-Enhanced)
    参数量 ~5.4M (取决于 hidden_dim)
    """

    def __init__(self,
                 in_features,
                 hidden_dim=96,  # 建议设为 96 或 128 以获得更强的拟合能力
                 drop=0.1):  # 为了降Loss，建议暂时把 dropout 设为 0 或极小
        super().__init__()

        self.norm = nn.LayerNorm(in_features)

        # Down
        self.down_proj = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()

        # Spatial Processing (核心升级)
        self.fusion = CoordinateFusion(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(drop)

        # Up
        self.up_proj = nn.Linear(hidden_dim, in_features)

        # LayerScale (收敛关键)
        self.layer_scale = LayerScale(in_features, init_values=1e-5)

        # 初始化
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, hw_shapes=None):
        identity = x

        # 1. Norm & Down
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)

        # 2. Spatial Process
        B, N, C = x.shape
        if hw_shapes is None:
            H = W = int(N ** 0.5)
        else:
            H, W = hw_shapes

        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)

        # 进入坐标感知融合
        x_spatial = self.fusion(x_spatial)

        x = x_spatial.flatten(2).transpose(1, 2)

        # 3. Up & Scale
        x = self.dropout(x)
        x = self.up_proj(x)

        # 4. LayerScale
        x = self.layer_scale(x)

        return identity + x


# ----------------------------------------------------
# 调试与参数量检查
# ----------------------------------------------------
if __name__ == "__main__":
    # 模拟环境
    bs, n, c = 2, 196, 768
    hw = (14, 14)
    x = torch.randn(bs, n, c)

    # 实例化 (hidden_dim=96 平衡参数量和性能)
    adapter = TBA_Adapter_v2(c, hidden_dim=96)

    # 前向传播
    y = adapter(x, hw)
    print(f"Output shape: {y.shape}")

    # 参数量统计
    # 假设有 24 个 Adapter (12层 * 2)
    single_params = sum(p.numel() for p in adapter.parameters())
    total_params = single_params * 24
    print(f"Single Adapter Params: {single_params / 1000:.2f} K")
    print(f"Total Params (Est. for 24 blocks): {total_params / 1e6:.2f} M")
    # 如果 96 导致参数略超，可以降到 80 或 88，
    # 但为了 Loss 降到 0.02，建议优先保持 96 并减少 dropout