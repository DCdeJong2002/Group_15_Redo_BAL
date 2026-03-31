import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile

# ============================================================
# 1. SCIENTIFIC PROCESSING (FOURIER SERIES METHOD WITH AVERAGING)
# ============================================================

def calculate_clean_fourier_spsl(p_mic, fs, segment_size=4096):
    """
    Calculates a clean SPSL [dB/Hz] line using averaged Fourier Series.
    Averaging segments reduces variance to prevent a 'wide noise area'.
    Equations followed:
    1. detrend(p) ensures a0/2 term is zero.
    2. df = fs / segment_size
    3. phi_segment = (an^2 + bn^2) / (2 * df)
    """
    p_ref = 20e-6
    num_segments = len(p_mic) // segment_size
    phi_accum = np.zeros(segment_size // 2 + 1)
    df = fs / segment_size
    
    for i in range(num_segments):
        # Extract segment and detrend (a0/2 = 0 per lecture slides)
        segment = p_mic[i*segment_size : (i+1)*segment_size]
        segment = segment - np.mean(segment)
        
        # Fourier Transform to get real an and imaginary bn coefficients
        X = np.fft.rfft(segment)
        an = (2.0 / segment_size) * np.real(X)
        bn = -(2.0 / segment_size) * np.imag(X)
        
        # Calculate Spectral Density phi(f) [Pa^2/Hz] for this segment
        phi_segment = (an**2 + bn**2) / (2.0 * df)
        phi_accum += phi_segment
        
    # Ensemble average segments to create the 'single line'
    phi_avg = phi_accum / num_segments
    
    # Convert Spectral Density to SPSL [dB/Hz]
    # Small constant added to avoid log10(0)
    spsl = 10 * np.log10(phi_avg / (p_ref**2) + 1e-12)
    freqs = np.fft.rfftfreq(segment_size, 1/fs)
    
    return freqs, spsl

def load_mic_data(folder_path, target_files, fs, D=0.2032):
    """Processes folder, calculates J, and applies Fourier analysis to each run."""
    results = []
    if not os.path.exists(folder_path): return pd.DataFrame()

    for meta_fn in target_files:
        meta_path = os.path.join(folder_path, meta_fn)
        if not os.path.exists(meta_path): continue
        
        df_meta = pd.read_csv(meta_path, sep=',', header=None)
        for _, row in df_meta.iterrows():
            # Extract metadata: Col 0=DPN, Col 6=V_inf, Col 12=AoA, Col 14=RPS
            dpn, v_inf, aoa, rps = int(float(row[0])), float(row[6]), round(float(row[12]), 1), float(row[14])
            
            # Calculate non-dimensional Advance Ratio J (critical for propeller loading)
            j_adv = round(v_inf / (rps * D), 2) if rps > 0 else 0
            
            tdms_fn = f"{meta_fn[:-4]}_run{dpn}_001.tdms"
            tdms_path = os.path.join(folder_path, tdms_fn)
            
            if os.path.exists(tdms_path):
                # Read TDMS binary acoustic data
                tdms_file = TdmsFile.read(tdms_path)
                p_mic = tdms_file.groups()[0].channels()[0].data
                
                # Perform the full Fourier transformation chain
                freqs, spsl = calculate_clean_fourier_spsl(p_mic, fs)
                results.append({
                    'config': meta_fn, 'aoa': aoa, 'j_adv': j_adv, 
                    'rps': rps, 'freqs': freqs, 'spsl': spsl, 'dpn': dpn
                })
    return pd.DataFrame(results)

# ============================================================
# 2. MAIN EXECUTION & SWEEP ANALYSIS
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mic_folder = os.path.join(script_dir, 'Mic')
    # Sampling frequency, diameter (from manual), and number of blades
    fs, D, num_blades = 51200.0, 0.2032, 6
    target_files = ['DPN18.txt', 'DPN19.txt', 'DPN26.txt', 'DPN27.txt']
    
    print("Executing Fourier Series analysis and J/AoA sweeps...")
    df = load_mic_data(mic_folder, target_files, fs, D)
    if df.empty: return

    # Define the custom bright color cycle requested by the user
    color_cycle = ['b', 'r', 'g'] # Bright Blue, Bright Red, Bright Green

    # --- PLOT 1: J Sweep - ABSOLUTE FREQUENCY (Constant AoA = 2.5°) ---
    # This plot visualizes how the noise severity changes as the propeller loading (J) varies.
    plt.figure(figsize=(9, 5))
    j_data = df[df['aoa'] == 2.5].sort_values('j_adv')
    
    j_colors = {1.6: 'blue', 2.0: 'red', 2.8: 'green'}

    for i, (_, row) in enumerate(j_data.iterrows()):
        # Identify the J-value and get the corresponding color
        current_j = round(row['j_adv'], 1)
        color = j_colors.get(current_j, 'black') # Defaults to black if J is unexpected
        
        # Calculate fundamental BPF
        bpf = row['rps'] * num_blades
        
        # 1. Plot the SPSL spectral line
        plt.plot(row['freqs'], row['spsl'], color=color, linewidth=1.2,
                 label=f"J = {row['j_adv']:.1f}")
        
        # 2. Plot the first four harmonics (n=1, 2, 3, 4)
        for n in range(1, 5):
            harmonic_freq = n * bpf
            
            # Label only the first BPF of each J value to keep the legend readable
            line_label = f"BPF (J={current_j}, 1st BPF ≈ {bpf:.0f}Hz)" if i < len(j_colors) and n == 1 else None
            
            plt.axvline(x=harmonic_freq, 
                        color=color, 
                        linestyle='--', 
                        alpha=0.3, 
                        linewidth=0.8,
                        label=line_label)
    
    '''plt.title("J-Sweep: Acoustic Severity vs. Propeller Loading (Constant AoA = 2.5°)")'''
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPSL [dB/Hz]")
    plt.xlim([0, 4000]) # Capture first ~5 harmonics at max RPS
    plt.ylim([35, 85])
    plt.legend(title="Advance ratio $J$", fontsize='small')
    plt.grid(True, alpha=0.3)

    # --- PLOT 2: AoA Sweep - NORMALIZED FREQUENCY (Constant J ≈ 1.6) ---
    # This plot isolates the "Propulsion-Airframe Integration Effect" by keeping the loading (J) constant.
    plt.figure(figsize=(9, 5))
    # Filter for data points very close to the target J=1.6
    aoa_data = df[abs(df['j_adv'] - 1.6) <= 0.05].sort_values('aoa')
    
    for i, (_, row) in enumerate(aoa_data.iterrows()):
        # Select the color from the cycle (blue, red, green)
        color = color_cycle[i % 3]
        
        # Normalize the frequency axis by the specific BPF of this run
        bpf = row['rps'] * num_blades
        plt.plot(row['freqs'] / bpf, row['spsl'], color=color, 
                 label=f"$\\alpha$ = {row['aoa']:.1f}°")
    
    '''plt.title("AoA Sweep: Propeller-Airframe Integration Effect (Constant J ≈ 1.6)")'''
    plt.xlabel("Normalized frequency [$f / BPF$]")
    plt.ylabel("SPSL [dB/Hz]")
    plt.xlim([0, 6]) # View fundamental through 6th harmonic
    plt.ylim([35, 85])
    plt.legend(title="Angle of Attack $\\alpha$", fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__": main()