import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.signal import welch

# ============================================================
# 1. SCIENTIFIC PROCESSING (WELCH METHOD - dB/Hz)
# ============================================================

def calculate_smooth_acoustics(p_mic, fs):
    """Calculates Sound Pressure Spectrum Level (SPSL) in [dB/Hz]."""
    p_ref = 20e-6
    # 4096 segments for a clean line
    freqs, psd = welch(p_mic, fs, nperseg=4096, window='hann', scaling='density')
    spsl = 10 * np.log10(psd / (p_ref**2))
    return freqs, spsl

def load_mic_data(folder_path, target_files, fs):
    """Loads and processes specific DPN metadata files from the Mic folder."""
    results = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        return pd.DataFrame()

    for meta_fn in target_files:
        meta_path = os.path.join(folder_path, meta_fn)
        if not os.path.exists(meta_path):
            continue
            
        try:
            df_meta = pd.read_csv(meta_path, sep=',', header=None)
            for _, row in df_meta.iterrows():
                dpn = int(float(row[0]))
                aoa = round(float(row[12]), 1)
                rps = round(float(row[14]), 1)
                
                tdms_fn = f"{meta_fn[:-4]}_run{dpn}_001.tdms"
                tdms_path = os.path.join(folder_path, tdms_fn)
                
                if os.path.exists(tdms_path):
                    tdms_file = TdmsFile.read(tdms_path)
                    group = tdms_file.groups()[0]
                    p_mic = group.channels()[0].data
                    
                    freqs, spsl = calculate_smooth_acoustics(p_mic, fs)
                    results.append({
                        'filename': meta_fn,
                        'aoa': aoa, 
                        'rps': rps, 
                        'freqs': freqs, 
                        'spsl': spsl, 
                        'dpn': dpn
                    })
        except Exception as e:
            print(f"Error processing {meta_fn}: {e}")
            
    return pd.DataFrame(results)

# ============================================================
# 2. MAIN EXECUTION & LINEAR PLOTTING
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mic_folder = os.path.join(script_dir, 'Mic')
    fs = 51200.0
    num_blades = 6
    
    # Specific files for studies
    target_files = ['DPN18.txt', 'DPN19.txt', 'DPN26.txt', 'DPN27.txt']
    
    print("Processing acoustics for linear frequency visualization...")
    df = load_mic_data(mic_folder, target_files, fs)

    if df.empty:
        print("No data found.")
        return

    # --- PLOT 1: RPS Sweep - RAW FREQUENCY (Constant AoA = 2.5) ---
    plt.figure(figsize=(10, 5))
    rps_data = df[(df['aoa'] == 2.5) & (df['filename'].isin(['DPN18.txt', 'DPN26.txt', 'DPN27.txt']))]
    rps_data = rps_data.sort_values('rps')

    for _, row in rps_data.iterrows():
        # Using standard plt.plot for Linear Scale
        plt.plot(row['freqs'], row['spsl'], label=f"RPS = {row['rps']:.1f} Hz")
    
    plt.title("RPS Sweep: Raw Linear Frequency (Constant AoA = 2.5°)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPSL [dB/Hz]")
    plt.xlim([0, 4000]) # Capture first ~5 harmonics at max RPS
    plt.ylim([0, 80])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- PLOT 2: RPS Sweep - NORMALIZED (Constant AoA = 2.5) ---
    plt.figure(figsize=(10, 5))
    for _, row in rps_data.iterrows():
        bpf = row['rps'] * num_blades
        plt.plot(row['freqs'] / bpf, row['spsl'], label=f"RPS = {row['rps']:.1f} Hz")
    
    plt.title("RPS Sweep: Normalized Linear Frequency (Constant AoA = 2.5°)")
    plt.xlabel("Normalized Frequency [$f / f_b$]")
    plt.ylabel("SPSL [dB/Hz]")
    plt.axvline(1, color='red', linestyle='--', alpha=0.5, label='BPF')
    plt.xlim([0, 6]) # Focus on 1st through 6th harmonics
    plt.ylim([0, 80])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- PLOT 3: AoA Sweep - NORMALIZED (Constant RPS ≈ 123 Hz) ---
    plt.figure(figsize=(10, 5))
    aoa_data = df[(abs(df['rps'] - 123.0) < 1.0) & (df['filename'].isin(['DPN18.txt', 'DPN19.txt']))]
    aoa_data = aoa_data.sort_values('aoa')

    for _, row in aoa_data.iterrows():
        bpf = row['rps'] * num_blades
        plt.plot(row['freqs'] / bpf, row['spsl'], label=f"AoA = {row['aoa']:.1f}°")
    
    plt.title("AoA Sweep: Normalized Linear Frequency (Constant RPS ≈ 123 Hz)")
    plt.xlabel("Normalized Frequency [$f / f_b$]")
    plt.ylabel("SPSL [dB/Hz]")
    plt.axvline(1, color='red', linestyle='--', alpha=0.5, label='BPF')
    plt.xlim([0, 6])
    plt.ylim([0, 80])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()