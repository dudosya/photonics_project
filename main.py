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
sampling_rate = 16*bit_rate  # should be about 4-16x of bit rate
num_bits = 1000
samples_per_bit = int(sampling_rate/bit_rate)
total_time = num_bits * bit_duration
time = np.linspace(0, total_time, num_bits * samples_per_bit)
mean = 0
stdev = 0.3
seed = 7
gaussian_var = 2



# run
data_stream = generate_random_data(num_bits ,  seed)
#print(f"Transmitted signal: {data_stream}")
waveform = generate_NRZ_waveform(data_stream, samples_per_bit , num_bits, mean, stdev, gaussian_var)
plot_NRZ_waveform(waveform)
output_bits = waveform2bits(waveform, samples_per_bit)
#print(f"Received signal:    {output_bits}")
ber = ber_calculator(data_stream, output_bits)
print(f"Bit Error Rate: {ber}%")
plot_eye_diagram(waveform, samples_per_bit, bit_rate, num_bits)



    