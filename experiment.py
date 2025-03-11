import numpy as np
import matplotlib.pyplot as plt

bit_rate = 200e6  
bit_duration = 1 / bit_rate  
num_bits = 100  
samples_per_bit = 100 
total_time = num_bits * bit_duration
time = np.linspace(0, total_time, num_bits * samples_per_bit)

# 0s and 1s
np.random.seed(42)  
data = np.random.randint(0, 2, num_bits)
print("Generated Data (first 20 bits):", data[:20])

# NRZ waveform 
waveform = np.zeros(len(time))
for i in range(num_bits):
    start_idx = i * samples_per_bit
    end_idx = (i + 1) * samples_per_bit
    waveform[start_idx:end_idx] = data[i] + np.random.normal(0,0.05)  # Rectangular pulse for NRZ

# low-pass filter effect
from scipy.ndimage import gaussian_filter1d

waveform = gaussian_filter1d(waveform, sigma=10) 

# the eye diagram
def plot_eye_diagram(waveform, samples_per_bit, bit_duration, num_bits):
    eye_duration = 2 * bit_duration
    eye_samples = 2 * samples_per_bit
    eye_time = np.linspace(-bit_duration, bit_duration, eye_samples)

    plt.figure(figsize=(10, 6))
    segments_plotted = 0 

    # цикл по уэвформу
    for i in range(0, num_bits - 2):  
        start_idx = i * samples_per_bit
        end_idx = start_idx + eye_samples
        if end_idx <= len(waveform):  
            segment = waveform[start_idx:end_idx]
            plt.plot(eye_time * 1e9, segment, 'b-', alpha=0.1)  # Time in ns, low alpha for overlap
            segments_plotted += 1

    print(f"Number of segments plotted: {segments_plotted}")
    plt.grid(True)
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title('Eye Diagram (NRZ Waveform)')
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)  
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)  
    plt.show()

plt.figure(figsize=(10, 4))
plt.plot(time * 1e9, waveform, 'b-', label='NRZ Waveform')  # Time in ns
plt.grid(True)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title('NRZ Waveform from Binary Data')
plt.legend()
plt.show()


plot_eye_diagram(waveform, samples_per_bit, bit_duration, num_bits)