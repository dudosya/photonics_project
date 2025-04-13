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
    data_stream = generate_random_data(num_bits, seed)
    plot_ber_vs_stdev(data_stream, samples_per_bit, num_bits, mean, stdev_list, gaussian_var)



def experiment_dispersion():
    data_stream = generate_random_data(num_bits, seed)
    dispersed_wave = generate_NRZ_waveform_with_dispersion(data_stream, bit_rate, samples_per_bit, None , L_km, D_ps_nm_km, lambda0_nm, stdev, gaussian_var)
    
    # True BER
    output_bits = waveform2bits(dispersed_wave, samples_per_bit)
    ber = ber_calculator(data_stream, output_bits)
    print(f"TRUE Bit Error Rate:      {ber}%")

    # Estim BER
    mean_zeros, mean_ones, stdev_zeros, stdev_ones = find_signal_stats(data_stream, dispersed_wave, samples_per_bit)
    q_factor = find_q_factor(mean_zeros, mean_ones, stdev_zeros, stdev_ones)
    estim_ber = ber_estimation(q_factor)
    print(f"ESTIMATED Bit Error Rate: {estim_ber:.3g}%")


if __name__ == "__main__":
    #experiment_eye_diagram()
    #experiment_ber_vs_stdev()
    experiment_dispersion()



    