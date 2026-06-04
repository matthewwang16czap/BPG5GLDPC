import torch
import torch.nn as nn
from sionna.phy.channel import AWGN, FlatFadingChannel


class BaseChannel(nn.Module):
    """Base class for AMC channels."""

    def forward(self, symbols: torch.Tensor, noise_var: float):
        """
        symbols  : (1, num_symbols) complex
        noise_var: scalar float
        Returns  : (rx, effective_noise_var)
        """
        raise NotImplementedError


class AWGNChannel(BaseChannel):
    def __init__(self, device="cpu"):
        super().__init__()
        self._channel = AWGN(device=str(device))

    def forward(self, symbols: torch.Tensor, noise_var: float):
        rx = self._channel(symbols, noise_var)
        return rx, noise_var / 2


class RayleighChannel(BaseChannel):
    def __init__(self, device="cpu"):
        super().__init__()
        self._channel = FlatFadingChannel(
            num_tx_ant=1,
            num_rx_ant=1,
            return_channel=True,
            device=str(device),
        )

    def forward(self, symbols: torch.Tensor, noise_var: float):
        num_symbols = symbols.shape[-1]
        flat = symbols.reshape(num_symbols, 1)       # (num_symbols, 1)

        rx_flat, h = self._channel(flat, noise_var)

        # Zero-forcing equalization
        h_sq  = h.squeeze()                          # (num_symbols,)
        h_mag = h_sq.abs().clamp(min=1e-8)
        rx_eq = rx_flat.squeeze() / h_sq             # (num_symbols,)

        # Effective noise variance after ZF: noise_var / |h|^2, averaged over block
        eff_noise_var = (noise_var / (h_mag ** 2)).mean().item()

        rx = rx_eq.reshape(1, num_symbols)
        return rx, eff_noise_var / 2


def build_channel(channel_type: str, device="cpu") -> BaseChannel:
    if channel_type == "awgn":
        return AWGNChannel(device=device)
    elif channel_type == "rayleigh":
        return RayleighChannel(device=device)
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")