import matplotlib.pyplot as plt
import numpy as np
from utils import *

def plot_NRZ_waveform(waveform):
    plt.grid(True)
    plt.xlabel("Samples") 
    plt.ylabel("Amplitude")
    plt.title("NRZ waveform at the receiver")
    plt.plot(waveform)
    plt.show()
    

def plot_eye_diagram(waveform, samples_per_bit , bit_rate, num_bits):
    eye_duration = 2 / bit_rate
    eye_samples =  2 * samples_per_bit
    eye_time = np.linspace(-1/bit_rate, 1/bit_rate, eye_samples)
    
    plt.figure(figsize=(10,6))
    segments_plotted = 0
    
    for i in range(0, num_bits-1):
        start_idx = i * samples_per_bit
        end_idx = start_idx + eye_samples
        if end_idx < len(waveform):
            segment = waveform[start_idx:end_idx]
            plt.plot(eye_time, segment, 'b-', alpha = 0.1)
            segments_plotted += 1
            
    print(f"Number of segments plotted: {segments_plotted}")
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Eye Diagram')  
    plt.show()


def plot_ber_vs_stdev(data_stream, samples_per_bit, num_bits, mean, stdev_list, gaussian_var):
    true_ber_list = []
    est_ber_list = []

    for each_stdev in stdev_list:
        waveform = generate_NRZ_waveform(data_stream, samples_per_bit, num_bits, mean, each_stdev, gaussian_var)

        # true ber
        output_bits = waveform2bits(waveform, samples_per_bit)
        ber = ber_calculator(data_stream, output_bits)
        true_ber_list.append(ber)

        # estim ber
        mean_zeros, mean_ones, stdev_zeros, stdev_ones = find_signal_stats(data_stream, waveform, samples_per_bit)
        q_factor = find_q_factor(mean_zeros, mean_ones, stdev_zeros, stdev_ones)
        ber_estim = ber_estimation(q_factor)
        est_ber_list.append(ber_estim)

    plt.figure(figsize=(10, 6))
    plt.plot(stdev_list, true_ber_list, marker='o', linestyle='-', color='blue', label='Measured BER')
    plt.plot(stdev_list, est_ber_list, marker='x', linestyle='--', color='red', label='Estimated BER (Q-factor)')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.ylabel("Bit Error Rate (BER)")
    plt.xlabel("Standard Deviation of Additive Gaussian Noise ($\sigma$)")
    plt.title("Measured vs. Estimated BER vs. Noise Standard Deviation")
    plt.legend()
    plt.tight_layout()
    plt.show()
