from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdapterCfg:
    embed_dims: int
    inner_dim: int = 64          # bottleneck
    scale: float = 0.1           # s
    dropout: float = 0.0
    learnable_scale: bool = False
    zero_init_up: bool = True


class AdaptFormer(nn.Module):
    """
    Residual bottleneck adapter that can be inserted anywhere in the block.
    Signature: forward(x, hw_shape=None) -> x'
    """
    def __init__(self, cfg: AdapterCfg):
        super().__init__()
        self.cfg = cfg

        self.down = nn.Linear(cfg.embed_dims, cfg.inner_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=cfg.dropout) if cfg.dropout and cfg.dropout > 0 else nn.Identity()
        self.up = nn.Linear(cfg.inner_dim, cfg.embed_dims, bias=True)

        if cfg.learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(cfg.scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(cfg.scale)), persistent=False)

        if cfg.zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, hw_shape=None) -> torch.Tensor:
        # hw_shape is intentionally ignored; kept only for API compatibility with Mona_PathoMSF
        y = self.down(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.up(y)
        return x + self.scale * y



