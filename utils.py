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

def chunk_list_comprehension(data, size):
  """Chunks a list into smaller lists of a specified size using list comprehension."""
  return [data[i : i + size] for i in range(0, len(data), size)]


def find_signal_stats(data_stream, waveform, samples_per_bit):
    # inputs: NRZ waveform, OG bits
    # output: stats i.e. 4 nums

    # examples:
    # data_stream = [0,1,0,1]
    # waveform: bunch of numbers

    # 1 waveform gets chopped to intervals
    waveform_list = chunk_list_comprehension(waveform, samples_per_bit)

    # 2 for each interval stats are found
    mean_wf_list = [np.mean(i) for i in waveform_list]
    stdev_wf_list = [np.std(i) for i in waveform_list]

    # 3 four lists are needed
    # means for 0s and 1s, stdevs for 0s and 1s
    mean_zeros_list = []
    mean_ones_list = []
    stdev_zeros_list = []
    stdev_ones_list = []

    for i,num in enumerate(data_stream):
        if num == 1:
            mean_ones_list.append(mean_wf_list[i])
            stdev_ones_list.append(stdev_wf_list[i])
        elif num == 0:
            mean_zeros_list.append(mean_wf_list[i]) 
            stdev_zeros_list.append(stdev_wf_list[i])
        else:
            print("There should be only 0s and 1s")

    mean_zeros = np.mean(mean_zeros_list)
    mean_ones = np.mean(mean_ones_list)
    stdev_zeros = np.mean(stdev_zeros_list)
    stdev_ones = np.mean(stdev_ones_list)

    return mean_zeros, mean_ones, stdev_zeros, stdev_ones
    
def find_q_factor(mean_zeros, mean_ones, stdev_zeros, stdev_ones):
    return (mean_ones - mean_zeros)/(stdev_ones+stdev_zeros)



from scipy.special import erfc 

def ber_estimation(q_factor):
    return 100*0.5 * erfc(q_factor/np.sqrt(2))








if __name__ == "__main__":
    from config import *
    data_stream = generate_random_data(num_bits)
    waveform = generate_NRZ_waveform(data_stream,samples_per_bit, num_bits, mean, stdev, gaussian_var)
    mean_zeros, mean_ones, stdev_zeros, stdev_ones = find_signal_stats(data_stream, waveform, samples_per_bit)
    q_factor = find_q_factor(mean_zeros, mean_ones, stdev_zeros, stdev_ones)
    ber_estimation = ber_estimation(q_factor)
    print(ber_estimation)
    
        
    


