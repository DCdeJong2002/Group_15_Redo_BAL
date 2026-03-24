import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq

# ============================================================
# 1. CORE PROCESSING FUNCTIONS
# ============================================================

def phase_avg_data(y, one_p, fs, rps, ph_intp, d_ph=0):
    """Segment data by rotation triggers and ensemble average."""
    n_s = len(one_p)
    t = np.linspace(0, n_s/fs, n_s, endpoint=False)
    peaks, _ = find_peaks(one_p, height=-0.0001)

    if len(peaks) < 2:
        return np.zeros(len(ph_intp)), np.array([])

    d_idx = int(round((d_ph / 360.0) * (1.0 / rps) * fs))
    peaks = peaks - d_idx
    peaks = peaks[peaks >= 0]
    
    rev_interpolated = []
    for j in range(len(peaks) - 1):
        y_rev = y[peaks[j]:peaks[j+1]+1]
        ph_rev = np.linspace(0, 2 * np.pi, len(y_rev))
        f_interp = interp1d(ph_rev, y_rev, kind='linear', fill_value="extrapolate")
        rev_interpolated.append(f_interp(ph_intp))

    y_avg = np.mean(np.array(rev_interpolated), axis=0)
    return y_avg, np.array(rev_interpolated)

def calculate_spl_spectrum(p_mic, fs):
    """FFT analysis for Sound Pressure Level in dB."""
    n = len(p_mic)
    p_fft = fft(p_mic)
    freqs = fftfreq(n, 1/fs)
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    psd = (np.abs(p_fft[pos_mask])**2) / (n * fs)
    p_ref = 20e-6 
    spl = 10 * np.log10(psd / (p_ref**2))
    return freqs, spl

# ============================================================
# 2. MAIN EXECUTION
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fn_folder = os.path.join(script_dir, 'Mic')
    
    fn_list = [f for f in os.listdir(fn_folder) if f.startswith('DPN') and f.endswith('.txt') and '_' not in f]
    fn_list.sort() 

    fs = 51200.0
    ph_intp = np.linspace(0, 2 * np.pi, 361)[:-1] 
    d_ph = 0

    all_config_results = {}

    for fn in fn_list:
        avg_path = os.path.join(fn_folder, fn)
        print(f"\n--- Processing Configuration: {fn} ---")
        
        try:
            avg_data = pd.read_csv(avg_path, sep=',', header=None).values
        except Exception as e:
            print(f" [!] Error reading {fn}: {e}")
            continue

        dpn_list = avg_data[:, 0].astype(int)
        aoa_list = avg_data[:, 12]
        rps_list = avg_data[:, 14]

        config_data = {'aoa': [], 'rps': [], 'p_rms': [], 'y_avg_samples': [], 'spectra': []}

        for j, dpn in enumerate(dpn_list):
            tdms_filename = f"{fn[:-4]}_run{dpn}_001.tdms"
            tdms_path = os.path.join(fn_folder, tdms_filename)

            if not os.path.exists(tdms_path):
                continue

            print(f" [+] Run {dpn}: AoA = {aoa_list[j]:.1f}°, RPS = {rps_list[j]:.1f} Hz")
            
            tdms_file = TdmsFile.read(tdms_path)
            group = tdms_file.groups()[0]
            p_mic = group.channels()[0].data
            one_p = group.channels()[1].data

            # Calculations
            y_avg, _ = phase_avg_data(p_mic, one_p, fs, rps_list[j], ph_intp, d_ph)
            freqs, spl = calculate_spl_spectrum(p_mic, fs)
            p_rms = np.sqrt(np.mean(y_avg**2))
            
            config_data['aoa'].append(aoa_list[j])
            config_data['rps'].append(rps_list[j])
            config_data['p_rms'].append(p_rms)
            
            # Save samples with RPS metadata
            if j == 0 or j == len(dpn_list)//2:
                config_data['y_avg_samples'].append((aoa_list[j], rps_list[j], y_avg))
                config_data['spectra'].append((aoa_list[j], rps_list[j], freqs, spl, rps_list[j]*6))

        all_config_results[fn] = config_data

    # ============================================================
    # 3. MULTI-CONFIG PLOTTING WITH RPS ANNOTATIONS
    # ============================================================
    if not all_config_results:
        return

    # FIGURE 1: Phase-Averaged Signal (Comparison in Radians)
    plt.figure(figsize=(10, 8))
    first_config = list(all_config_results.keys())[0]
    samples = all_config_results[first_config]['y_avg_samples']
    for i, (aoa, rps, y_val) in enumerate(samples[:2]):
        plt.subplot(2, 1, i+1)
        plt.plot(ph_intp, y_val, color='tab:blue')
        plt.title(f"{first_config} | Signal at AoA = {aoa:.1f}°, RPS = {rps:.1f} Hz")
        plt.ylabel("Pressure [Pa]")
        plt.xlabel("Phase Angle [rad]")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # FIGURE 2: Spectral Comparison (SPL vs Frequency)
    plt.figure(figsize=(10, 6))
    for config_name, data in all_config_results.items():
        if data['spectra']:
            aoa, rps, f, s, bpf = data['spectra'][0]
            plt.semilogx(f, s, label=f"{config_name} (AoA={aoa:.1f}°, {rps:.1f} Hz)", alpha=0.7)
            plt.axvline(bpf, color='red', linestyle='--', alpha=0.2)
    plt.title("Sound Pressure Level (SPL) Comparison with BPF Markers")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB/Hz]")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, which="both", alpha=0.2)

    # FIGURE 3: Comparative RMS Trend (Labeling points with RPS)
    plt.figure(figsize=(10, 6))
    for config_name, data in all_config_results.items():
        plt.plot(data['aoa'], data['p_rms'], 'o-', label=config_name)
        # Add RPS text annotations to individual points
        for k in range(len(data['aoa'])):
            plt.text(data['aoa'][k], data['p_rms'][k], f" {data['rps'][k]:.0f}Hz", 
                     fontsize=8, verticalalignment='bottom')

    plt.title("Acoustic Intensity Trend: All Experimental Configurations")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Tonal pRMS [Pa]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()