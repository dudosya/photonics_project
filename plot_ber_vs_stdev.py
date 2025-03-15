# plot how BER increases as stdev increases
# generate list of stdevs 
# for each stdev waveform is generated
# from each waveform waveform2bits function is applied
# ber is then calculated and then added to ber_results list
# in the end, ber vs stdev is plotted

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt


    from generate_random_data import *
    from generate_NRZ_waveform import *
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
    stdev = 3
    seed = 7
    gaussian_var = 2

    data_stream = generate_random_data(num_bits ,  seed)

    stdev_list = np.arange(0.1, 3, 0.1)
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

