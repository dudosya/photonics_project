from config import *
from plot import *
from utils import *



def main():
    data_stream = generate_random_data(num_bits, seed)
    plot_ber_vs_stdev(data_stream, samples_per_bit, num_bits, mean, stdev_list, gaussian_var)
    waveform = generate_NRZ_waveform_nonlinear(data_stream, bit_rate, samples_per_bit, sampling_rate, avg_power_mW, 
                                                         L_km, alpha_db_km, D_ps_nm_km, gamma_W_km, lambda0_nm, num_steps,
                                                         mean, stdev, gaussian_var)
    plot_NRZ_waveform(waveform, samples_per_bit)
    plot_eye_diagram(waveform, samples_per_bit, bit_rate, num_bits)

    # Estim BER
    mean_zeros, mean_ones, stdev_zeros, stdev_ones = find_signal_stats(data_stream, waveform, samples_per_bit)
    q_factor = find_q_factor(mean_zeros, mean_ones, stdev_zeros, stdev_ones)
    estim_ber = ber_estimation(q_factor)
    print(f"ESTIMATED Bit Error Rate: {estim_ber:.3g}%")
    

if __name__ == "__main__":   
    main()