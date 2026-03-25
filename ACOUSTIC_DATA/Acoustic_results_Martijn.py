import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile

# ============================================================
# 1. SCIENTIFIC PROCESSING (FOURIER SERIES AVERAGING)
# ============================================================

def calculate_clean_fourier_spsl(p_mic, fs, segment_size=4096):
    """
    Calculates a clean SPSL [dB/Hz] line using the Fourier Series method.
    Uses averaging to prevent 'wide noise area' while keeping an, bn math.
    """
    p_ref = 20e-6
    # Determine number of segments for averaging to get a 'single line'
    num_segments = len(p_mic) // segment_size
    phi_accum = np.zeros(segment_size // 2 + 1)
    df = fs / segment_size
    
    for i in range(num_segments):
        # Extract segment and remove DC offset (image_97f3e2.png: a0/2 = 0)
        segment = p_mic[i*segment_size : (i+1)*segment_size]
        segment = segment - np.mean(segment)
        
        # Fourier Transform to get an, bn
        X = np.fft.rfft(segment)
        an = (2.0 / segment_size) * np.real(X)
        bn = -(2.0 / segment_size) * np.imag(X)
        
        # Calculate Spectral Density phi(f) (image_97f401.png)
        phi_segment = (an**2 + bn**2) / (2.0 * df)
        phi_accum += phi_segment
        
    # Ensemble Average to get the 'clean line'
    phi_avg = phi_accum / num_segments
    
    # Convert to SPSL [dB/Hz] (image_97f426.png)
    spsl = 10 * np.log10(phi_avg / (p_ref**2) + 1e-12)
    freqs = np.fft.rfftfreq(segment_size, 1/fs)
    
    return freqs, spsl

def load_mic_data(folder_path, target_files, fs, D=0.2032):
    """Processes folder and calculates Advance Ratio (J)."""
    results = []
    if not os.path.exists(folder_path): return pd.DataFrame()

    for meta_fn in target_files:
        meta_path = os.path.join(folder_path, meta_fn)
        if not os.path.exists(meta_path): continue
        
        df_meta = pd.read_csv(meta_path, sep=',', header=None)
        for _, row in df_meta.iterrows():
            dpn, v_inf, aoa, rps = int(float(row[0])), float(row[6]), round(float(row[12]), 1), float(row[14])
            j_adv = round(v_inf / (rps * D), 2) if rps > 0 else 0
            
            tdms_fn = f"{meta_fn[:-4]}_run{dpn}_001.tdms"
            tdms_path = os.path.join(folder_path, tdms_fn)
            
            if os.path.exists(tdms_path):
                tdms_file = TdmsFile.read(tdms_path)
                p_mic = tdms_file.groups()[0].channels()[0].data
                
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
    fs, D, num_blades = 51200.0, 0.2032, 6
    target_files = ['DPN18.txt', 'DPN19.txt', 'DPN26.txt', 'DPN27.txt']
    
    df = load_mic_data(mic_folder, target_files, fs, D)
    if df.empty: return

    # --- SWEEP 1: J Sweep (Constant AoA = 2.5°) ---
    plt.figure(figsize=(10, 5))
    j_data = df[df['aoa'] == 2.5].sort_values('j_adv')
    for _, row in j_data.iterrows():
        plt.plot(row['freqs'], row['spsl'], label=f"J = {row['j_adv']:.2f}")
    
    plt.title("J-Sweep: Acoustic Severity vs. Propeller Loading (AoA = 2.5°)")
    plt.xlabel("Frequency [Hz]"); plt.ylabel("SPSL [dB/Hz]")
    plt.xlim([0, 5000]); plt.ylim([35, 85]); plt.legend(title="Advance Ratio"); plt.grid(True, alpha=0.3)

    # --- SWEEP 2: AoA Sweep (Constant J ≈ 1.6) ---
    plt.figure(figsize=(10, 5))
    aoa_data = df[abs(df['j_adv'] - 1.6) <= 0.05].sort_values('aoa')
    for _, row in aoa_data.iterrows():
        bpf = row['rps'] * num_blades
        plt.plot(row['freqs'] / bpf, row['spsl'], label=f"AoA = {row['aoa']:.1f}°")
    
    plt.title("AoA Sweep: Propeller-Airframe Interaction (Constant J ≈ 1.6)")
    plt.xlabel("Normalized Frequency [$f / f_b$]"); plt.ylabel("SPSL [dB/Hz]")
    plt.axvline(1, color='red', linestyle='--', alpha=0.5, label='BPF')
    plt.xlim([0, 6]); plt.ylim([35, 85]); plt.legend(title="Angle of Attack"); plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__": main()