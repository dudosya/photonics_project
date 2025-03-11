import matplotlib.pyplot as plt

#for testing
from generate_random_data import *
from generate_NRZ_waveform import *

def plot_NRZ_waveform(waveform):
    plt.grid(True)
    plt.xlabel("Samples") 
    plt.ylabel("Amplitude")
    plt.title("NRZ waveform at the receiver")
    plt.plot(waveform)
    plt.show()
    

if __name__ == "__main__":
    random_data = generate_random_data(10)
    waveform = generate_NRZ_waveform(random_data, num_bits=10)
    plot_NRZ_waveform(waveform)
