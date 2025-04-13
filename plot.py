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
    ber_list = []

    for each_stdev in stdev_list:
        waveform = generate_NRZ_waveform(data_stream, samples_per_bit, num_bits, mean, each_stdev, gaussian_var)
        output_bits = waveform2bits(waveform, samples_per_bit)
        ber = ber_calculator(data_stream, output_bits)
        ber_list.append(ber)

    plt.grid(True)
    plt.ylabel("Bit Error Rate (%)")
    plt.xlabel("Stdev of Gaussian Noise")
    plt.plot(stdev_list, ber_list)
    plt.title("BER vs. Stdev graph")
    plt.show()