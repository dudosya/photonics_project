import numpy as np


# config
bit_rate = 10e9
bit_duration = 1 / bit_rate
sampling_rate = 16*bit_rate  # should be about 4-16x of bit rate
num_bits = 1000
samples_per_bit = int(sampling_rate/bit_rate)
total_time = num_bits * bit_duration
time = np.linspace(0, total_time, num_bits * samples_per_bit)
mean = 0
stdev = 0.05
seed = 7
gaussian_var = 2.5

# for ber vs stdev
stdev_list = np.arange(0.1, 3, 0.1)

# for dispersion
L_km = 10
D_ps_nm_km = 17
lambda0_nm = 1550
