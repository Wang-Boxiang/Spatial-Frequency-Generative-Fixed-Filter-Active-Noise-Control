import numpy as np


def construct_filter(sub_filters: np.ndarray, weights: np.ndarray):
    """Combine sub-control filters using the CRNN regression weights."""

    weights = np.expand_dims(weights, axis=0)
    reconstructed_filter = np.matmul(weights, sub_filters)
    return reconstructed_filter, weights


def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to a signal at the requested SNR."""

    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise


def ambeo_vr_mic_positions(
    center=(0, 0, 0),
    radius: float = 0.0125,
    azimuths_deg=(0, 180, 90, 270),
    elevations_deg=(35.264, 35.264, -35.264, -35.264),
) -> np.ndarray:
    """Return the four AMBEO-style microphone positions around a center point."""

    azimuths = np.deg2rad(azimuths_deg)
    elevations = np.deg2rad(elevations_deg)
    positions = []

    for azimuth, elevation in zip(azimuths, elevations):
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        positions.append([x, y, z])

    return np.array(positions) + np.asarray(center)
