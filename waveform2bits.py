import numpy as np

def waveform2bits(waveform, samples_per_bit):
    output_bits = []
    
    for i in range(int(len(waveform)/samples_per_bit)):
        mean_num = np.mean(waveform[i*samples_per_bit:(i+1)*samples_per_bit])
        if  mean_num > 0.5:
            output_bits.append(1)
        elif mean_num < 0.5:
            output_bits.append(0)
            
    return np.array(output_bits)
    
        