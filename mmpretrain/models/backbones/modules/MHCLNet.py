import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from mmcv.cnn import BaseModule  # type: ignore
except Exception:
    BaseModule = nn.Module


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob <= 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_mask


class DWAnisotropic(nn.Module):
    """Depthwise (1 x k) + (k x 1) anisotropic convolution."""

    def __init__(self, channels: int, k: int = 7):
        super().__init__()
        pad = k // 2
        self.conv_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, k),
            padding=(0, pad),
            groups=channels,
            bias=False,
        )
        self.conv_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k, 1),
            padding=(pad, 0),
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_v(self.conv_h(x))


class AdaptiveFusion(nn.Module):
    """Channel-wise adaptive fusion used inside MHFF."""

    def __init__(self, channels: int, num_branches: int, reduction: int = 4, tau: float = 1.5):
        super().__init__()
        self.channels = channels
        self.num_branches = num_branches
        self.tau = float(tau)

        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels * num_branches, bias=True)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.base_logits = nn.Parameter(torch.zeros(num_branches, channels))
        self.ctx_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats) == self.num_branches
        b, c, _, _ = feats[0].shape

        u = feats[0]
        for i in range(1, self.num_branches):
            u = u + feats[i]
        s = u.mean(dim=(2, 3))
        s = F.layer_norm(s, (c,))

        z = self.act(self.fc1(s))
        delta = self.fc2(z).view(b, self.num_branches, c)

        logits = self.base_logits.unsqueeze(0) + self.ctx_scale * delta
        attn = torch.softmax(logits / self.tau, dim=1)
        attn = attn.unsqueeze(-1).unsqueeze(-1)
        return [attn[:, i] for i in range(self.num_branches)]


class MHFF(BaseModule):
    """Morphology-aware Histopathology Feature Fusion."""

    def __init__(
        self,
        channels: int,
        dilation: int = 2,
        aniso_k: int = 7,
        progressive: bool = True,
        tau: float = 1.5,
    ):
        super().__init__()
        self.progressive = bool(progressive)

        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)

        self.dwaniso = DWAnisotropic(channels, k=aniso_k)
        self.dwdil = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.alpha_aniso = nn.Parameter(torch.tensor(0.0))
        self.alpha_dil = nn.Parameter(torch.tensor(0.0))

        self.alpha_me = nn.Parameter(torch.tensor(0.0))
        self.adaptive_fusion = AdaptiveFusion(channels, num_branches=3, reduction=4, tau=tau)

    @staticmethod
    def _highpass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.progressive:
            y3 = self.dw3(x)
            y5 = self.dw5(x + y3)
            y7_base = self.dw7(x + y5)
        else:
            y3 = self.dw3(x)
            y5 = self.dw5(x)
            y7_base = self.dw7(x)

        y7 = y7_base + self.alpha_aniso * self.dwaniso(x) + self.alpha_dil * self.dwdil(x)

        w3, w5, w7 = self.adaptive_fusion([y3, y5, y7])
        fused = y3 * w3 + y5 * w5 + y7 * w7

        out = fused + identity
        out = out + self.alpha_me * self._highpass(identity)
        return out


class HCCR(BaseModule):
    """Hierarchical Channel Context Refinement."""

    def __init__(self, channels: int, mixer_rank: Optional[int] = None):
        super().__init__()
        hidden = mixer_rank if mixer_rank is not None else max(8, channels // 8)

        # Conv1 / Conv2 in the paper.
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        # GLU-style projection.
        self.glu_proj = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual: Conv1 -> GeLU -> Conv2
        x = x + self.conv2(F.gelu(self.conv1(x)))
        
        identity = x
        
        # Second residual: GLU gating
        a, b = self.glu_proj(x).chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        
        x = identity + x
        
        return F.gelu(x)


class MHCLCore(BaseModule):
    """Core conv-space modeling stage: MHFF followed by HCCR."""

    def __init__(
        self,
        channels: int,
        dilation: int = 2,
        aniso_k: int = 7,
        progressive: bool = True,
        tau: float = 1.5,
        mixer_rank: Optional[int] = None,
    ):
        super().__init__()
        self.mhff = MHFF(
            channels=channels,
            dilation=dilation,
            aniso_k=aniso_k,
            progressive=progressive,
            tau=tau,
        )
        self.hccr = HCCR(channels=channels, mixer_rank=mixer_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mhff(x)
        x = self.hccr(x)
        return x


class MHCLAdapter(BaseModule):
    """Token-space adapter with paper-aligned naming: MHFF + HCCR."""

    def __init__(
        self,
        in_dim: int,
        inner_dim: int = 64,
        drop: float = 0.1,
        dilation: int = 2,
        aniso_k: int = 7,
        progressive: bool = True,
        tau: float = 1.5,
        use_norm_gating: bool = True,
        mixer_rank: Optional[int] = None,
        drop_path_prob: Optional[float] = None,
    ):
        super().__init__()
        self.use_norm_gating = bool(use_norm_gating)

        # Linear (C -> r) and Linear (r -> C) in the paper.
        self.linear_down = nn.Linear(in_dim, inner_dim)
        self.linear_up = nn.Linear(inner_dim, in_dim)
        self.dropout = nn.Dropout(p=drop)

        self.core = MHCLCore(
            channels=inner_dim,
            dilation=dilation,
            aniso_k=aniso_k,
            progressive=progressive,
            tau=tau,
            mixer_rank=mixer_rank,
        )

        self.norm = nn.LayerNorm(in_dim)
        self.lambda_ = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.lambda_x = nn.Parameter(torch.ones(in_dim))

        if drop_path_prob is None:
            drop_path_prob = min(0.1, float(drop) * 0.5)
        self.drop_path_prob = float(drop_path_prob)

    def forward(self, x: torch.Tensor, hw_shapes=None) -> torch.Tensor:
        identity = x

        if self.use_norm_gating:
            x = self.norm(x) * self.lambda_ + x * self.lambda_x

        y = self.linear_down(x)
        b, n, c = y.shape
        h, w = hw_shapes
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)

        y = self.core(y)

        y = y.permute(0, 2, 3, 1).reshape(b, n, c)
        y = self.dropout(y)
        y = self.linear_up(y)
        y = drop_path(y, drop_prob=self.drop_path_prob, training=self.training)
        return identity + y

