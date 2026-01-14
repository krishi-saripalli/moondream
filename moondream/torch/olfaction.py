import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Tuple
from PIL import Image

from .layers import attn, layer_norm, mlp
from .image_crops import overlap_crop_image
from .config import OlfactionConfig


def padding_mask(smells: torch.Tensor):
    B, T, D = smells.shape
    mask = smells.abs().sum(dim=-1) > 0
    assert smells.shape == (B, T)
    return mask


def pad_to_max_timestep(smell: torch.Tensor, config: OlfactionConfig) -> torch.Tensor:
    T, D = smell.shape
    total_padding = min(T - config.max_timestep, 0)
    padded = F.pad(
        smell,
        [0, 0, total_padding // 2, total_padding - (total_padding // 2)],
        mode="constant",
        value=0.0,
    )
    assert padded.shape == (config.max_timestep, D)
    return padded


def concat_sample_and_ambient(
    sample: torch.Tensor, ambient: torch.Tensor
) -> torch.Tensor:
    T, D = sample.shape
    concated = torch.cat([sample, ambient], dim=0)
    assert concated.shape == (2 * T, D)
    return concated


def olfaction_encoder(input_BTD: torch.Tensor, w: nn.Module, config: OlfactionConfig):
    B, T, D = input_BTD.shape
    # TODO: verify that this correctly prevents any padded timesteps from contributing
    mask = padding_mask(input_BTD).broadcast_to(B, 1, T, T)
    x = w.in_proj(input_BTD)
    x = x + w.pos_emb
    for block in w.blocks:
        x = x + attn(
            layer_norm(x, block.ln1), block.attn, n_heads=config.enc_n_heads, mask=mask
        )
        x = x + mlp(layer_norm(x, block.ln2), block.mlp)
    x = layer_norm(x, w.post_ln)
    return x


def olfaction_projection(
    encoded: torch.Tensor,
    w: nn.Module,
    config: OlfactionConfig,
):
    return mlp(encoded, w.out_proj_mlp)


def build_olfaction_model(config: OlfactionConfig, dtype: torch.dtype):
    num_timesteps = 2 * config.max_timestep
    olfaction = nn.ModuleDict(
        {
            "in_proj": nn.Linear(config.smell_dim, config.enc_dim),
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(
                                        config.enc_dim, 3 * config.enc_dim, dtype=dtype
                                    ),
                                    "proj": nn.Linear(
                                        config.enc_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                            "ln2": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(
                                        config.enc_dim, config.enc_ff_dim, dtype=dtype
                                    ),
                                    "fc2": nn.Linear(
                                        config.enc_ff_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                        }
                    )
                    for _ in range(config.enc_n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.enc_dim, dtype=dtype),
            "out_proj_mlp": nn.ModuleDict(
                {
                    "fc1": nn.Linear(
                        config.enc_dim, config.proj_inner_dim, dtype=dtype
                    ),
                    "fc2": nn.Linear(
                        config.proj_inner_dim, config.out_dim, dtype=dtype
                    ),
                }
            ),
        }
    )

    olfaction.pos_emb = nn.Parameter(
        torch.zeros((1, num_timesteps, config.enc_dim), dtype=dtype)
    )

    return olfaction
