import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mmcv.cnn import BaseModule  # type: ignore
except Exception:
    BaseModule = nn.Module

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob <= 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_mask


class DWAnisotropic(nn.Module):
    def __init__(self, channels: int, k: int = 7):
        super().__init__()
        pad = k // 2
        self.h = nn.Conv2d(
            channels, channels,
            kernel_size=(1, k),
            padding=(0, pad),
            groups=channels,
            bias=False,
        )
        self.v = nn.Conv2d(
            channels, channels,
            kernel_size=(k, 1),
            padding=(pad, 0),
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.h(x))


class SKChannelSelectorHybrid(nn.Module):

    def __init__(self, channels: int, num_branches: int, reduction: int = 4, tau: float = 1.5):
        super().__init__()
        self.channels = channels
        self.num_branches = num_branches
        self.tau = float(tau)

        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels * num_branches, bias=True)

        # Zero-init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # Weight-decay robust bias term
        self.base_logits = nn.Parameter(torch.zeros(num_branches, channels))
        self.ctx_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, feats):
        assert len(feats) == self.num_branches
        B, C, _, _ = feats[0].shape

        u = feats[0]
        for i in range(1, self.num_branches):
            u = u + feats[i]
        s = u.mean(dim=(2, 3))             # [B, C]
        s = F.layer_norm(s, (C,))

        z = self.act(self.fc1(s))
        delta = self.fc2(z).view(B, self.num_branches, C)  # [B, K, C]

        logits = self.base_logits.unsqueeze(0) + self.ctx_scale * delta
        attn = torch.softmax(logits / self.tau, dim=1)
        attn = attn.unsqueeze(-1).unsqueeze(-1)             # [B, K, C, 1, 1]
        return [attn[:, i] for i in range(self.num_branches)]


class MonaOp_PathoMSF(BaseModule):
    def __init__(
        self,
        in_features: int,
        dilation: int = 2,
        aniso_k: int = 7,
        progressive: bool = True,
        tau: float = 1.5,
        mixer_rank: Optional[int] = None,
    ):
        super().__init__()
        C = in_features
        self.progressive = bool(progressive)

        # Core isotropic branches
        self.dw3 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.dw5 = nn.Conv2d(C, C, kernel_size=5, padding=2, groups=C, bias=False)
        self.dw7 = nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C, bias=False)

        # Bonus paths for the 7x7 branch
        self.dwaniso = DWAnisotropic(C, k=aniso_k)
        self.dwdil = nn.Conv2d(
            C, C, kernel_size=3,
            padding=dilation, dilation=dilation,
            groups=C, bias=False
        )
        self.alpha_aniso = nn.Parameter(torch.tensor(0.0))
        self.alpha_dil = nn.Parameter(torch.tensor(0.0))

        self.alpha_hp = nn.Parameter(torch.tensor(0.0))

        # Selector (3 branches)
        self.K = 3
        self.selector = SKChannelSelectorHybrid(C, num_branches=self.K, reduction=4, tau=tau)

        # Low-rank mixer in conv space (cheap channel interaction)
        r = mixer_rank if mixer_rank is not None else max(8, C // 8)
        self.pw_down = nn.Conv2d(C, r, kernel_size=1, bias=True)
        self.pw_up = nn.Conv2d(r, C, kernel_size=1, bias=True)
        # Zero-init up-proj -> mixer starts as 0 (safe)
        nn.init.zeros_(self.pw_up.weight)
        nn.init.zeros_(self.pw_up.bias)

        # GLU projector for expressive but stable gating
        self.projector = nn.Conv2d(C, C * 2, kernel_size=1, bias=True)

    @staticmethod
    def _highpass(x: torch.Tensor) -> torch.Tensor:
        # Local mean subtraction (3x3). Padding keeps shape.
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

        feats = [y3, y5, y7]
        w3, w5, w7 = self.selector(feats)

        fused = y3 * w3 + y5 * w5 + y7 * w7
        x = fused + identity

        # Add high-frequency residual (learned strength)
        x = x + self.alpha_hp * self._highpass(identity)

        # Low-rank channel mixing (residual)
        x = x + self.pw_up(F.gelu(self.pw_down(x)))

        # GLU residual
        identity2 = x
        a, b = self.projector(x).chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        return identity2 + x


class Mona_PathoMSF(BaseModule):

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

        self.project1 = nn.Linear(in_dim, inner_dim)
        self.project2 = nn.Linear(inner_dim, in_dim)
        self.dropout = nn.Dropout(p=drop)

        self.adapter_conv = MonaOp_PathoMSF(
            inner_dim,
            dilation=dilation,
            aniso_k=aniso_k,
            progressive=progressive,
            tau=tau,
            mixer_rank=mixer_rank,
        )

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

        # Default: modest stochastic depth from drop (no config change needed)
        if drop_path_prob is None:
            drop_path_prob = min(0.1, float(drop) * 0.5)
        self.drop_path_prob = float(drop_path_prob)

    def forward(self, x: torch.Tensor, hw_shapes=None) -> torch.Tensor:
        identity = x

        if self.use_norm_gating:
            x = self.norm(x) * self.gamma + x * self.gammax

        y = self.project1(x)  # [B, N, C]
        b, n, c = y.shape
        h, w = hw_shapes
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]

        y = self.adapter_conv(y)

        y = y.permute(0, 2, 3, 1).reshape(b, n, c)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.project2(y)

        y = drop_path(y, drop_prob=self.drop_path_prob, training=self.training)
        return identity + y


def _sanity():
    m = Mona_PathoMSF(in_dim=768, inner_dim=64)
    x = torch.randn(2, 196, 768)
    y = m(x, (14, 14))
    return y.shape

if __name__ == "__main__":
    print(_sanity())
