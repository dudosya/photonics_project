import numpy as np

from generate_random_data import *
from generate_NRZ_waveform import *
from plot_eye_diagram import *
from plot_NRZ_waveform import *
from waveform2bits import *
from ber import *


# config
bit_rate = 1e6
bit_duration = 1 / bit_rate
sampling_rate = 10*bit_rate  # should be about 4-16x of bit rate
num_bits = 100
samples_per_bit = int(sampling_rate/bit_rate)
total_time = num_bits * bit_duration
time = np.linspace(0, total_time, num_bits * samples_per_bit)
mean = 0
stdev = 0.01
seed = 5
gaussian_var = 0.7



# run
data_stream = generate_random_data(num_bits, seed)
waveform = generate_NRZ_waveform(data_stream, samples_per_bit, num_bits, mean, stdev, gaussian_var )
output_bits = waveform2bits(waveform, samples_per_bit)
ber = ber_calculator(data_stream, output_bits)
print(f"Bit Error Rate: {ber}%")
plot_NRZ_waveform(waveform)
plot_eye_diagram(waveform, samples_per_bit, bit_rate)



    