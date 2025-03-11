import numpy as np

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