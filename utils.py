import numpy as np
from scipy.ndimage import gaussian_filter1d

def generate_random_data(num_bits=100, seed = 7):
    """generate binary data stream

    Args:
        num_bits (int): number of bits to be generated. 
        seed (int, optional): seed for reproducibility. 

    Returns:
        list: list of binary data of length num_bits
    """
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.randint(0,2,num_bits)
    else:
        return np.random.randint(0,2, num_bits)
    

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


def ber_calculator(original_data, received_data):
    assert len(original_data) == len(received_data)
    num_errors = 0
    
    for i in range(len(original_data)):
        if original_data[i] != received_data[i]:  
            num_errors += 1
    
    return 100*num_errors/len(original_data)



def waveform2bits(waveform, samples_per_bit):
    output_bits = []
    
    for i in range(int(len(waveform)/samples_per_bit)):
        mean_num = np.mean(waveform[i*samples_per_bit:(i+1)*samples_per_bit])
        if  mean_num > 0.5:
            output_bits.append(1)
        elif mean_num < 0.5:
            output_bits.append(0)
            
    return np.array(output_bits)
    
        
    


