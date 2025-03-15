import numpy as np
from scipy.ndimage import gaussian_filter1d

def generate_NRZ_waveform(data_stream, samples_per_bit=100, num_bits=100, mean=0, stdev=0.01, gaussian_var=2):
    """
    Generates an NRZ waveform with added Gaussian noise and filtering.

    Args:
        data_stream: A list or NumPy array of binary data (0s and 1s).
        samples_per_bit: The number of samples to represent each bit.
        num_bits: The total number of bits (should match the length of data_stream).
        mean: The mean of the Gaussian noise.
        stdev: The standard deviation of the Gaussian noise.
        gaussian_var: The standard deviation (sigma) for the Gaussian filter.

    Returns:
        A NumPy array representing the noisy, filtered NRZ waveform.
    """

    if len(data_stream) != num_bits:
        raise ValueError("Length of data_stream must equal num_bits")

    nrz = np.zeros(samples_per_bit * num_bits)

    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = (i + 1) * samples_per_bit

        # Correctly add noise to the entire segment for this bit.
        # Use vectorized operations for speed and clarity.
        nrz[start_idx:end_idx] = data_stream[i] + np.random.normal(mean, stdev, size=samples_per_bit)

    waveform = gaussian_filter1d(nrz, sigma=gaussian_var)
    return waveform

