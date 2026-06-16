import torch


def compute_stft_for_allchannels(
    signal_nch: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 64,
) -> torch.Tensor:
    """Return concatenated magnitude and phase STFT maps for a multi-channel signal."""

    window = torch.hann_window(
        n_fft,
        device=signal_nch.device,
        dtype=signal_nch.dtype,
    )

    magnitude_maps = []
    phase_maps = []
    for channel in range(signal_nch.shape[0]):
        stft = torch.stft(
            signal_nch[channel],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        magnitude_maps.append(torch.abs(stft).unsqueeze(0))
        phase_maps.append(torch.angle(stft).unsqueeze(0))

    return torch.cat(
        [
            torch.cat(magnitude_maps, dim=0),
            torch.cat(phase_maps, dim=0),
        ],
        dim=0,
    )
