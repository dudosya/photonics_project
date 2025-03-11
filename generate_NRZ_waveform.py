import numpy as np
from scipy.ndimage import gaussian_filter1d


def generate_NRZ_waveform(data_stream, samples_per_bit = 100, num_bits = 100, mean = 0, stdev = 0.01, gaussian_var = 10):
    nrz = np.zeros(samples_per_bit * num_bits)
    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = (i+1) * samples_per_bit
        nrz[start_idx:end_idx] = data_stream[i] + np.random.normal(mean, stdev)
    
    waveform = gaussian_filter1d(nrz, sigma = gaussian_var)
    
    
    return waveform 