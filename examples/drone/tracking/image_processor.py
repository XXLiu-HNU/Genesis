import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------
# Utils
# ---------------------------
def _make_gaussian_kernel(k: int, sigma: Optional[float] = None, device=None, dtype=torch.float32):
    assert k % 2 == 1 and k > 1, "kernel size must be odd and >1"
    if sigma is None:
        # heuristics: similar to OpenCV default
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1, 1)
    return g, g.transpose(2, 3)


def _gaussian_blur_2d(x: torch.Tensor, k: int, sigma: Optional[float] = None) -> torch.Tensor:
    if k <= 1:
        return x
    g_h, g_w = _make_gaussian_kernel(k, sigma, device=x.device, dtype=x.dtype)
    # depthwise separable blur
    x = F.conv2d(x, g_h, padding=(k // 2, 0), groups=x.size(1))
    x = F.conv2d(x, g_w, padding=(0, k // 2), groups=x.size(1))
    return x


def _dilate_bool_mask(mask_bool: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return mask_bool
    m = (~mask_bool).float().unsqueeze(1)  # invalid as 1
    m = F.max_pool2d(F.pad(m, (k // 2, k // 2, k // 2, k // 2), mode="replicate"),
                     kernel_size=k, stride=1)
    return ~(m.squeeze(1) > 0.5)  # back to bool valid mask


def normalize_depth(depths: torch.Tensor, max_range: float, mode: str = "log") -> torch.Tensor:
    if mode == "linear":
        x = torch.clamp(depths, 0.0, max_range) / float(max_range)
    elif mode == "inv":
        x = 1.0 / torch.clamp(depths, 1e-3, max_range)  # inverse depth
        x = x / x.max().clamp(min=1e-6)
    else:  # "log"
        x = torch.log1p(torch.clamp(depths, 0.0, max_range)) / torch.log1p(torch.tensor(max_range, device=depths.device, dtype=depths.dtype))
    return x


# ---------------------------
# Augmentations
# ---------------------------
def apply_depth_augmentations(
    depths: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    max_range: float = 20.0,
    min_range: float = 0.0,
    add_gaussian_std: float = 0.005,      # relative to max_range if relative_sigma=True
    mult_std: float = 0.01,               # per-frame multiplicative noise (global)
    lowfreq_mult_strength: float = 0.02,  # low-frequency multiplicative field strength
    p_hole: float = 0.002,                # per-pixel holes
    hole_block_prob: float = 0.08,        # probability to add 1 block/stripe occlusion
    block_max_frac: float = 0.25,         # up to 25% of H/W (works better on 48x64)
    stripe_prob: float = 0.15,            # add row/col banding
    stripe_axis: Optional[str] = None,    # 'row'/'col'/None; None=random
    stripe_amp_frac: float = 0.02,        # ~2% of max_range
    quantize_bits: Optional[int] = None,  # e.g. 12; None to disable
    quantize_jitter: bool = True,         # de-quantization dither within 0..1 LSB
    gaussian_blur_kernel: int = 3,        # true gaussian (odd); 0/1 disables
    seed: Optional[int] = None,
    relative_sigma: bool = True,
    dilate_invalid: int = 2,              # dilate invalid (no-hit) border by N pixels
    soft_mask: bool = True,               # produce 0.5 on dilated boundary
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    depths: (N,H,W) or (H,W) meters. no-hit encoded as max_range.
    Returns: (aug_depths (N,H,W), mask (N,H,W) bool valid hits)
    """
    if device is None:
        device = depths.device
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    x = depths.clone().to(device)

    # ensure batch dim
    added_batch = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        added_batch = True

    N, H, W = x.shape

    # clamp & base mask
    x = torch.clamp(x, min_range, max_range)
    mask = (x < max_range)  # valid hits

    if training:
        # Global multiplicative (per frame)
        if mult_std and mult_std > 0:
            eps = torch.normal(0.0, mult_std, size=(N, 1, 1), device=device, generator=g)
            x = x * (1.0 + eps)

        # Low-frequency multiplicative field
        if lowfreq_mult_strength and lowfreq_mult_strength > 0:
            lf = torch.normal(0.0, 1.0, size=(N, 1, H, W), device=device, generator=g)
            lf = _gaussian_blur_2d(lf, k=9)  # smooth to low frequency
            lf = lf / (lf.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
            x = x * (1.0 + lowfreq_mult_strength * lf.squeeze(1))

        # Additive gaussian noise
        if add_gaussian_std and add_gaussian_std > 0:
            sigma = add_gaussian_std * max_range if relative_sigma else add_gaussian_std
            noise = torch.normal(0.0, sigma, size=x.shape, device=device, generator=g)
            x = x + noise

        # Pixel holes
        if p_hole and p_hole > 0:
            hole_mask = torch.rand((N, H, W), device=device, generator=g) < p_hole
            x[hole_mask] = max_range
            mask[hole_mask] = False

        # Block/stripe occlusions
        if hole_block_prob and hole_block_prob > 0:
            probs = torch.rand((N,), device=device, generator=g)
            for i in range(N):
                if probs[i] < hole_block_prob:
                    # choose block or stripe
                    use_stripe = (torch.rand((), generator=g, device=device) < 0.5)
                    if use_stripe:
                        # random horizontal or vertical stripe
                        axis = stripe_axis or ("row" if torch.rand((), generator=g, device=device) < 0.5 else "col")
                        if axis == "row":
                            h = max(1, int(torch.randint(2, max(3, int(H * block_max_frac)), (1,), generator=g).item()))
                            y0 = torch.randint(0, max(1, H - h + 1), (1,), generator=g).item()
                            x[i, y0:y0+h, :] = max_range
                            mask[i, y0:y0+h, :] = False
                        else:
                            w = max(1, int(torch.randint(2, max(3, int(W * block_max_frac)), (1,), generator=g).item()))
                            x0 = torch.randint(0, max(1, W - w + 1), (1,), generator=g).item()
                            x[i, :, x0:x0+w] = max_range
                            mask[i, :, x0:x0+w] = False
                    else:
                        # rectangular block
                        fh = max(1, int(torch.randint(2, max(3, int(H * block_max_frac)), (1,), generator=g).item()))
                        fw = max(1, int(torch.randint(2, max(3, int(W * block_max_frac)), (1,), generator=g).item()))
                        y0 = torch.randint(0, max(1, H - fh + 1), (1,), generator=g).item()
                        x0 = torch.randint(0, max(1, W - fw + 1), (1,), generator=g).item()
                        x[i, y0:y0+fh, x0:x0+fw] = max_range
                        mask[i, y0:y0+fh, x0:x0+fw] = False

        # Banding noise (sinusoidal along rows or cols)
        if stripe_prob and torch.rand((), generator=g, device=device) < stripe_prob:
            axis = stripe_axis or ("row" if torch.rand((), generator=g, device=device) < 0.5 else "col")
            amp = stripe_amp_frac * max_range
            if axis == "row":
                yy = torch.arange(H, device=device).float().view(1, H, 1)
                phase = 2 * torch.pi * torch.rand((), generator=g, device=device)
                freq = torch.rand((), generator=g, device=device) * 0.15 + 0.05
                band = amp * torch.sin(freq * yy + phase)
                x = x + band
            else:
                xx = torch.arange(W, device=device).float().view(1, 1, W)
                phase = 2 * torch.pi * torch.rand((), generator=g, device=device)
                freq = torch.rand((), generator=g, device=device) * 0.15 + 0.05
                band = amp * torch.sin(freq * xx + phase)
                x = x + band

        # Quantization + dither
        if quantize_bits is not None and quantize_bits > 1:
            levels = 2 ** quantize_bits - 1
            x = torch.clamp(x, min_range, max_range)
            x = torch.round((x - min_range) / (max_range - min_range) * levels)
            if quantize_jitter:
                # add 0..1 LSB uniform noise before de-quantize
                x = x + torch.rand_like(x, generator=g).clamp(0, 1)
                x = x.clamp(0, levels)
            x = x / levels * (max_range - min_range) + min_range

        # True Gaussian blur (optional)
        if gaussian_blur_kernel and gaussian_blur_kernel > 1:
            xp = x.unsqueeze(1)  # (N,1,H,W)
            xp = _gaussian_blur_2d(xp, gaussian_blur_kernel)
            x = xp.squeeze(1)

    # recompute / dilate mask
    mask = (x < max_range)
    if dilate_invalid and dilate_invalid > 0:
        mask_dil = _dilate_bool_mask(mask, k=int(dilate_invalid) * 2 + 1)
        if soft_mask:
            # 0: invalid, 0.5: boundary (dilated-invalid but valid in base), 1: valid
            boundary = (~mask) & mask_dil
            # keep boolean mask as "hard valid" for return; soft mask用于输入通道
            soft = mask.float()
            soft[boundary] = 0.5
            mask_soft = soft
        else:
            mask_soft = mask.float()
        mask_for_return = mask  # bool
    else:
        mask_soft = mask.float()
        mask_for_return = mask

    if added_batch:
        x = x.squeeze(0)
        mask_for_return = mask_for_return.squeeze(0)
        mask_soft = mask_soft.squeeze(0)

    return x, mask_for_return, mask_soft  # (aug_depths, hard_mask_bool, soft_mask_float)


# ---------------------------
# Encoder
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=c)

    def forward(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.relu(x + h)


class SmallDepthEncoder(nn.Module):
    """(B, C, H, W) -> (B, out_dim)"""
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.gn1 = nn.GroupNorm(8, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.res = ResidualBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.res(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ---------------------------
# Processor
# ---------------------------
class ImageProcessor:
    """Augment depth maps and produce feature embeddings."""
    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_range: float = 20.0,
        encoder_out_dim: int = 128,
        use_mask_channel: bool = True,
        use_soft_mask: bool = True,
        quantize_bits: Optional[int] = None,
        norm_mode: str = "log",  # 'log' | 'linear' | 'inv'
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_range = max_range
        self.use_mask_channel = use_mask_channel
        self.use_soft_mask = use_soft_mask
        self.quantize_bits = quantize_bits
        self.norm_mode = norm_mode

        in_ch = 1 + (1 if use_mask_channel else 0)
        self.encoder = SmallDepthEncoder(in_channels=in_ch, out_dim=encoder_out_dim).to(self.device)

    def process(
        self,
        depths: torch.Tensor,
        *,
        training: bool = True,
        augment_params: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            depths: (N,H,W) or (H,W) meters. no-hit expected as max_range.
        Returns:
            features: (N, feat_dim)
            depths_aug: (N,H,W) meters
            mask: (N,H,W) bool valid hits
        """
        if augment_params is None:
            augment_params = dict(
                max_range=self.max_range,
                add_gaussian_std=0.005,
                mult_std=0.01,
                lowfreq_mult_strength=0.02,
                p_hole=0.002,
                hole_block_prob=0.10,
                block_max_frac=0.25,
                stripe_prob=0.15,
                stripe_axis=None,
                stripe_amp_frac=0.02,
                quantize_bits=self.quantize_bits,
                quantize_jitter=True,
                gaussian_blur_kernel=3,
                relative_sigma=True,
                dilate_invalid=2,
                soft_mask=self.use_soft_mask,
                training=training,
            )

        depths = depths.to(self.device)

        if seed is not None:
            augment_params = dict(augment_params)
            augment_params["seed"] = int(seed)

        depths_aug, mask_hard, mask_soft = apply_depth_augmentations(depths, device=self.device, **augment_params)

        # normalize depth
        inp = normalize_depth(depths_aug, self.max_range, mode=self.norm_mode)

        # add channel dim
        inp = inp.unsqueeze(1)  # (N,1,H,W)
        if self.use_mask_channel:
            mask_chan = (mask_soft if self.use_soft_mask else mask_hard.float()).unsqueeze(1)
            encoder_input = torch.cat([inp, mask_chan.to(dtype=inp.dtype)], dim=1)
        else:
            encoder_input = inp

        feats = self.encoder(encoder_input)
        return feats, depths_aug, mask_hard



if __name__ == "__main__":
    # quick smoke test
    device = torch.device("cpu")
    proc = ImageProcessor(device=device, max_range=20.0, encoder_out_dim=128, use_mask_channel=True, use_soft_mask=True)
    depths = torch.rand((4, 48, 64), dtype=torch.float32) * 10.0  # synthetic
    feats, depths_aug, mask = proc.process(depths, training=True, seed=42)
    print("feats", feats.shape)            # e.g. (4, 128)
    print("depths_aug", depths_aug.shape)  # (4, 48, 64)
    print("mask", mask.shape, mask.dtype)  # (4, 48, 64) torch.bool
