import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft, fftfreq
from scipy.constants import c, pi
from scipy.constants import k as boltzmann_k
from scipy.special import erfc 

def generate_random_data(num_bits=100, seed = 7):
    """generate binary data stream

    Args:
        num_bits (int): number of bits to be generated. 
        seed (int, optional): seed for reproducibility. 

    Returns:
        list: list of binary data of length num_bits
    """
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.randint(0,2,num_bits)
    else:
        return np.random.randint(0,2, num_bits)
    

def generate_NRZ_waveform(data_stream, samples_per_bit=100, num_bits=100, mean=0, stdev=0.01, gaussian_var=2):
    """
    Generates an NRZ waveform with added Gaussian noise and filtering.

    Args:
        data_stream: A list or NumPy array of binary data (0s and 1s).
        samples_per_bit: The number of samples to represent each bit.
        num_bits: The total number of bits (should match the length of data_stream).
        mean: The mean of the Gaussian noise.
        stdev: The standard deviation of the Gaussian noise.
        gaussian_var: The standard deviation (sigma) for the Gaussian filter.

    Returns:
        A NumPy array representing the noisy, filtered NRZ waveform.
    """

    if len(data_stream) != num_bits:
        raise ValueError("Length of data_stream must equal num_bits")

    nrz = np.zeros(samples_per_bit * num_bits)

    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = (i + 1) * samples_per_bit

        # Correctly add noise to the entire segment for this bit.
        # Use vectorized operations for speed and clarity.
        nrz[start_idx:end_idx] = data_stream[i] + np.random.normal(mean, stdev, size=samples_per_bit)

    waveform = gaussian_filter1d(nrz, sigma=gaussian_var)
    return waveform


def apply_chromatic_dispersion(signal, fs, L_m, D_si, lambda0_m):
    """
    Applies chromatic dispersion to a time-domain signal using frequency domain filtering.

    Args:
        signal (np.array): Input ideal time-domain signal (before noise/filtering).
        fs (float): Sampling frequency (Hz).
        L_m (float): Fiber length (meters).
        D_si (float): Dispersion parameter (s/m^2 - SI units!).
                      Example conversion: D [ps/(nm*km)] * 1e-6 -> D [s/m^2]
        lambda0_m (float): Center wavelength (meters).

    Returns:
        np.array: Time-domain signal with dispersion applied.
    """
    N = len(signal)

    # --- Calculate Beta2 (GVD parameter) ---
    # beta2 = - (D * lambda0^2) / (2 * pi * c)  Units: s^2/m
    beta2 = - (D_si * lambda0_m**2) / (2 * pi * c)

    # --- Frequency Domain Operations ---
    signal_fft = fft(signal)
    freqs_hz = fftfreq(N, d=1/fs)  # Frequency vector (Hz)
    omega = 2 * pi * freqs_hz      # Angular frequency vector (rad/s)

    # --- Dispersion Transfer Function Phase ---
    # Phase Shift = - (beta2 / 2) * omega^2 * L
    # Using +j here because signal processing FFT often defines phase this way
    # for a filter implementing a delay/phase shift.
    dispersion_phase = - (beta2 / 2) * (omega**2) * L_m
    H = np.exp(1j * dispersion_phase)

    # --- Apply Filter and Inverse Transform ---
    dispersed_fft = signal_fft * H
    dispersed_signal = ifft(dispersed_fft)

    # Return the real part (imaginary part should be negligible)
    return np.real(dispersed_signal)


# 2. Change the NRZ waveform generation function
def generate_NRZ_waveform_with_dispersion(
    data_stream,
    bit_rate,
    samples_per_bit=10, # Reduced default, adjust as needed
    sampling_rate=None, # Calculate if not given
    L_km=50,            # Fiber length in km
    D_ps_nm_km=17,      # Dispersion parameter in ps/(nm*km)
    lambda0_nm=1550,    # Wavelength in nm
    mean=0,             # Noise mean
    stdev=0.01,         # Noise standard deviation
    gaussian_var=2      # Sigma for Gaussian *receiver* filter
    ):
    """
    Generates an NRZ waveform, applies chromatic dispersion, adds Gaussian noise,
    and applies a final Gaussian filter (receiver filter).

    Args:
        data_stream: List or NumPy array of binary data (0s and 1s).
        bit_rate: The bit rate in bits per second (e.g., 10e9 for 10 Gbps).
        samples_per_bit: Number of samples per bit.
        sampling_rate: Sampling rate (Hz). If None, calculated from bit_rate
                       and samples_per_bit.
        L_km: Fiber length in kilometers.
        D_ps_nm_km: Chromatic dispersion parameter in ps/(nm*km).
        lambda0_nm: Center wavelength in nanometers.
        mean: Mean of the additive Gaussian noise.
        stdev: Standard deviation of the additive Gaussian noise.
        gaussian_var: Sigma for the Gaussian filter applied *after* noise
                      (simulates receiver bandwidth).

    Returns:
        A NumPy array representing the final waveform.
    """
    num_bits = len(data_stream)

    if sampling_rate is None:
        sampling_rate = bit_rate * samples_per_bit
    elif samples_per_bit != int(round(sampling_rate / bit_rate)):
         # Recalculate samples_per_bit if sampling_rate is specified and doesn't match
         samples_per_bit = int(round(sampling_rate / bit_rate))
         print(f"Recalculated samples_per_bit = {samples_per_bit} based on sampling_rate and bit_rate")

    if samples_per_bit < 2:
        raise ValueError("samples_per_bit must be >= 2 for meaningful simulation.")

    total_samples = samples_per_bit * num_bits

    # --- Generate Ideal NRZ waveform (+1/-1 levels often better for physics) ---
    # Use np.repeat for speed
    ideal_nrz = np.repeat(np.array(data_stream) * 2 - 1, samples_per_bit) # Maps 0->-1, 1->+1

    # --- Prepare Parameters for Dispersion Function (Convert to SI Units) ---
    L_m = L_km * 1000.0
    lambda0_m = lambda0_nm * 1e-9
    # Convert D from ps/(nm*km) to s/m^2
    # D [s/m^2] = D [ps/nm/km] * (1e-12 / (1e-9 * 1e3)) = D [ps/nm/km] * 1e-6
    D_si = D_ps_nm_km * 1e-6

    # --- Apply Chromatic Dispersion ---
    dispersed_nrz = apply_chromatic_dispersion(
        ideal_nrz,
        sampling_rate,
        L_m,
        D_si,
        lambda0_m
    )

    # --- Add Gaussian Noise ---
    # Generate noise for the entire waveform length
    noise = np.random.normal(mean, stdev, size=total_samples)
    noisy_dispersed_nrz = dispersed_nrz + noise

    # --- Apply Receiver Gaussian Filter ---
    # This filter simulates the limited bandwidth of the receiver
    final_waveform = gaussian_filter1d(noisy_dispersed_nrz, sigma=gaussian_var)

    return final_waveform




def ber_calculator(original_data, received_data):
    assert len(original_data) == len(received_data)
    num_errors = 0
    
    for i in range(len(original_data)):
        if original_data[i] != received_data[i]:  
            num_errors += 1
    
    return 100*num_errors/len(original_data)



def waveform2bits(waveform, samples_per_bit):
    output_bits = []
    
    for i in range(int(len(waveform)/samples_per_bit)):
        mean_num = np.mean(waveform[i*samples_per_bit:(i+1)*samples_per_bit])
        if  mean_num > 0.5:
            output_bits.append(1)
        else:
            output_bits.append(0)
            
    return np.array(output_bits)

def chunk_list_comprehension(data, size):
  """Chunks a list into smaller lists of a specified size using list comprehension."""
  return [data[i : i + size] for i in range(0, len(data), size)]


def find_signal_stats(data_stream, waveform, samples_per_bit):
    # inputs: NRZ waveform, OG bits
    # output: stats i.e. 4 nums

    # examples:
    # data_stream = [0,1,0,1]
    # waveform: bunch of numbers

    # 1 waveform gets chopped to intervals
    waveform_list = chunk_list_comprehension(waveform, samples_per_bit)

    # 2 for each interval stats are found
    mean_wf_list = [np.mean(i) for i in waveform_list]
    stdev_wf_list = [np.std(i) for i in waveform_list]

    # 3 four lists are needed
    # means for 0s and 1s, stdevs for 0s and 1s
    mean_zeros_list = []
    mean_ones_list = []
    stdev_zeros_list = []
    stdev_ones_list = []

    for i,num in enumerate(data_stream):
        if num == 1:
            mean_ones_list.append(mean_wf_list[i])
            stdev_ones_list.append(stdev_wf_list[i])
        elif num == 0:
            mean_zeros_list.append(mean_wf_list[i]) 
            stdev_zeros_list.append(stdev_wf_list[i])
        else:
            print("There should be only 0s and 1s")

    mean_zeros = np.mean(mean_zeros_list)
    mean_ones = np.mean(mean_ones_list)
    stdev_zeros = np.mean(stdev_zeros_list)
    stdev_ones = np.mean(stdev_ones_list)

    return mean_zeros, mean_ones, stdev_zeros, stdev_ones
    
def find_q_factor(mean_zeros, mean_ones, stdev_zeros, stdev_ones):
    return (mean_ones - mean_zeros)/(stdev_ones+stdev_zeros)





def ber_estimation(q_factor):
    return 100*0.5 * erfc(q_factor/np.sqrt(2))



# CONVERSION FUNCS

def dbkm_to_npm(alpha_db_km):
    """Converts attenuation from dB/km to Nepers/m."""
    # alpha (Np/m) = alpha (dB/km) / (10 * log10(e) * 1000)
    # 10 * log10(e) is approx 4.343
    return alpha_db_km / (4.342944819 * 1000.0)

def ps_nm_km_to_si(D_ps_nm_km):
    """Converts dispersion parameter D from ps/(nm*km) to s/m^2."""
    # D [s/m^2] = D [ps/nm/km] * (1e-12 / (1e-9 * 1e3)) = D [ps/nm/km] * 1e-6
    return D_ps_nm_km * 1e-6

def gamma_W_km_to_si(gamma_W_km):
    """Converts non-linear parameter gamma from 1/(W*km) to 1/(W*m)."""
    return gamma_W_km / 1000.0

def calculate_beta2_si(D_si, lambda0_m):
    """Calculates beta2 (GVD) in s^2/m from D (in s/m^2) and lambda0 (in m)."""
    if D_si is None or lambda0_m is None:
        return 0.0 # No dispersion if D is not provided
    # beta2 = - (D * lambda0^2) / (2 * pi * c)  Units: s^2/m
    return - (D_si * lambda0_m**2) / (2 * pi * c)

# --- Main Non-Linear Waveform Generation Function ---

def generate_NRZ_waveform_nonlinear(
    data_stream,
    bit_rate,
    samples_per_bit=16, # Usually needs more samples for NLSE
    sampling_rate=None, # Calculate if not given
    avg_power_mW=1.0,   # Average input power in mW
    L_km=50,            # Fiber length in kilometers
    alpha_db_km=0.2,    # Attenuation coefficient in dB/km
    D_ps_nm_km=17,      # Dispersion parameter in ps/(nm*km)
    gamma_W_km=1.3,     # Non-linear parameter in 1/(W*km)
    lambda0_nm=1550,    # Center wavelength in nanometers
    num_steps=100,      # Number of steps for SSFM
    mean=0,             # Receiver noise mean
    stdev=0.01,         # Receiver noise standard deviation (relative to peak field maybe?)
    gaussian_var=2      # Sigma for Gaussian *receiver* filter
    ):
    """
    Generates an NRZ waveform, simulates propagation with dispersion, loss,
    and Kerr non-linearity (SPM) using SSFM, adds receiver noise, and applies
    a final Gaussian filter.

    Args:
        data_stream: List or NumPy array of binary data (0s and 1s).
        bit_rate: The bit rate in bits per second (e.g., 10e9 for 10 Gbps).
        samples_per_bit: Number of samples per bit. Needs to be sufficient.
        sampling_rate: Sampling rate (Hz). If None, calculated.
        avg_power_mW: Average optical power launched into the fiber (mW).
        L_km: Fiber length in kilometers.
        alpha_db_km: Fiber attenuation coefficient in dB/km.
        D_ps_nm_km: Chromatic dispersion parameter in ps/(nm*km).
        gamma_W_km: Non-linear Kerr parameter in 1/(W*km).
        lambda0_nm: Center wavelength in nanometers.
        num_steps: Number of steps for the Split-Step Fourier Method.
        mean: Mean of the additive Gaussian receiver noise field.
        stdev: Standard deviation of the additive Gaussian receiver noise field.
               Interpretation depends on normalization. Assumed relative to field.
        gaussian_var: Sigma for the Gaussian filter applied *after* noise
                      (simulates receiver bandwidth).

    Returns:
        A NumPy array representing the final waveform (optical field amplitude).
        Note: The output is complex before noise/filtering if needed. Here we return real part.
    """
    num_bits = len(data_stream)
    if sampling_rate is None:
        sampling_rate = bit_rate * samples_per_bit
    elif samples_per_bit != int(round(sampling_rate / bit_rate)):
         samples_per_bit = int(round(sampling_rate / bit_rate))
         print(f"Recalculated samples_per_bit = {samples_per_bit} based on sampling_rate and bit_rate")

    if samples_per_bit < 8: # Non-linear sims usually need more resolution
        print(f"Warning: samples_per_bit ({samples_per_bit}) is low for non-linear simulation.")

    total_samples = samples_per_bit * num_bits
    time_step = 1.0 / sampling_rate
    time_vector = np.arange(0, total_samples * time_step, time_step)

    # --- Generate Ideal NRZ waveform and scale to optical field amplitude ---
    # Assume 50% mark density for avg power -> peak power calculation
    # PeakPower ~ 2 * AvgPower for NRZ
    # Amplitude ~ sqrt(PeakPower)
    P_avg_W = avg_power_mW * 1e-3
    P_peak_W = 2.0 * P_avg_W # Approximation for NRZ
    amplitude_scale = np.sqrt(P_peak_W)

    # Map 0 -> 0, 1 -> amplitude_scale (representing field, not power)
    ideal_nrz_field = np.repeat(np.array(data_stream) * amplitude_scale, samples_per_bit)

    # --- Prepare Physical Parameters in SI Units ---
    L_m = L_km * 1000.0
    lambda0_m = lambda0_nm * 1e-9
    alpha_npm = dbkm_to_npm(alpha_db_km)
    D_si = ps_nm_km_to_si(D_ps_nm_km)
    beta2_si = calculate_beta2_si(D_si, lambda0_m)
    gamma_si = gamma_W_km_to_si(gamma_W_km) # Units: 1/(W*m)

    # --- Prepare for SSFM ---
    dz = L_m / num_steps          # Length of each step
    omega = 2 * pi * fftfreq(total_samples, d=time_step) # Angular frequencies

    # Linear Operator (Frequency Domain for one step dz)
    # Includes dispersion and loss
    # H_linear = exp( (j*beta2/2 * omega^2 - alpha/2) * dz ) <-- sign convention can vary
    # Let's use the convention from previous dispersion function for consistency:
    linear_operator = np.exp(1j * (beta2_si / 2) * (omega**2) * dz - (alpha_npm / 2) * dz)

    # --- Execute SSFM ---
    current_field = ideal_nrz_field.astype(complex) # Start with ideal field, ensure complex type

    for i in range(num_steps):
        # Apply Non-Linear Operator (Time Domain)
        # Phase shift = gamma * |A|^2 * dz
        nonlinear_phase_shift = gamma_si * np.abs(current_field)**2 * dz
        field_after_NL = current_field * np.exp(1j * nonlinear_phase_shift)

        # Apply Linear Operator (Frequency Domain)
        field_fft = fft(field_after_NL)
        field_fft_after_L = field_fft * linear_operator
        current_field = ifft(field_fft_after_L)

    # current_field now holds the optical field after fiber propagation A(t, z=L)

    # --- Add Receiver Noise (Additive Gaussian Noise on the field) ---
    # Generate noise with the specified std dev.
    # Note: This assumes stdev relates to the field amplitude.
    # More complex noise models (shot, thermal) would scale differently.
    noise_field = np.random.normal(mean, stdev, size=total_samples) \
                + 1j * np.random.normal(mean, stdev, size=total_samples) # Complex noise for field
    noisy_field = current_field + noise_field

    # --- Apply Receiver Gaussian Filter ---
    # Filter the real and imaginary parts separately if needed, or filter magnitude.
    # Usually, photodiode detects power |A|^2. Filtering happens post-detection.
    # For simplicity here, we filter the complex field (less physically direct).
    # A more realistic model filters the detected power signal.
    # Let's return the real part after filtering for similarity to previous function.
    filtered_real = gaussian_filter1d(np.real(noisy_field), sigma=gaussian_var)
    # filtered_imag = gaussian_filter1d(np.imag(noisy_field), sigma=gaussian_var)
    # final_waveform = filtered_real + 1j*filtered_imag # Keep complex if needed later

    return filtered_real # Return real part to mimic previous function output type

# --- Example Usage ---
if __name__ == '__main__':
    # Simulation Parameters
    bits_to_send = 10 # More bits often needed to see NL effects build up
    my_bit_rate = 10e9       # 10 Gbps
    my_samples_per_bit = 32 # Increased samples per bit

    # Generate random bits
    my_data = np.random.randint(0, 2, bits_to_send)

    # Generate waveform with dispersion & non-linearity
    final_wave_nl = generate_NRZ_waveform_nonlinear(
        data_stream=my_data,
        bit_rate=my_bit_rate,
        samples_per_bit=my_samples_per_bit,
        avg_power_mW=2.0,   # Slightly higher power to see NL effects
        L_km=80,            # Longer distance
        alpha_db_km=0.2,
        D_ps_nm_km=17,
        gamma_W_km=1.3,
        lambda0_nm=1550,
        num_steps=150,      # More steps for longer distance/accuracy
        stdev=0.03,         # Receiver noise level
        gaussian_var=3.0    # Receiver filter sigma
    )

    print(f"Generated non-linear waveform with {len(final_wave_nl)} samples.")

    # Optional: Plot waveform segment
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        # Plot magnitude or real part? Real part for now.
        plt.plot(final_wave_nl[:my_samples_per_bit*10]) # Plot first 10 bits
        plt.title("First 10 Bits of Final Waveform (Non-Linear)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude (Real Part)")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Cannot plot example waveform.")
        
    


