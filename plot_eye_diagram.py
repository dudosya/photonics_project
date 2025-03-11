import matplotlib.pyplot as plt
import numpy as np


def plot_eye_diagram(waveform, samples_per_bit = 100, bit_rate = 200e6, num_bits = 100):
    eye_duration = 2 / bit_rate
    eye_samples =  3 * samples_per_bit
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