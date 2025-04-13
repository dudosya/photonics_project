from config import *
from plot import *
from utils import *



def experiment_eye_diagram():
    data_stream = generate_random_data(num_bits ,  seed)
    #print(f"Transmitted signal: {data_stream}")
    waveform = generate_NRZ_waveform(data_stream, samples_per_bit , num_bits, mean, stdev, gaussian_var)
    # plot_NRZ_waveform(waveform)
    output_bits = waveform2bits(waveform, samples_per_bit)
    #print(f"Received signal:    {output_bits}")
    ber = ber_calculator(data_stream, output_bits)
    print(f"TRUE Bit Error Rate: {ber}%")
    plot_eye_diagram(waveform, samples_per_bit, bit_rate, num_bits)


def experiment_ber_vs_stdev():
    data_stream = generate_random_data(num_bits ,  seed)
    plot_ber_vs_stdev(data_stream, samples_per_bit, num_bits, mean, stdev_list, gaussian_var)



if __name__ == "__main__":
    experiment_eye_diagram()
    #experiment_ber_vs_stdev()



    