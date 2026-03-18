import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# ============================================================
# 1. CORE FUNCTION: Phase Averaging
# ============================================================
def phase_avg_data(y, one_p, fs, rps, ph_intp, d_ph=0):
    """
    Performs averaging of measurement data over rotation using 1P trigger.
    Equivalent to phaseAvgData.m
    """
    n_s = len(one_p)
    t = np.linspace(0, n_s/fs, n_s, endpoint=False)

    # Find indices of trigger peaks (Rising edges)
    # MATLAB uses -0.0001 as a threshold for your specific sensor
    peaks, _ = find_peaks(one_p, height=-0.0001)

    if len(peaks) < 2:
        print("Warning: 1P trigger signal does not include enough edges.")
        return np.zeros(len(ph_intp)), 0

    # Calculate RPS from trigger to verify stability
    t_1p = t[peaks]
    rps_1p = 1.0 / np.diff(t_1p)
    
    if np.max(rps_1p) > 1.1 * rps or np.min(rps_1p) < 0.9 * rps:
        print(f"Warning: Computed RPS ({np.mean(rps_1p):.2f}) deviates significantly from motor RPS.")

    # Determine offset
    d_idx_sig_1p = int(round((d_ph / 360.0) * (1.0 / rps) * fs))
    peaks = peaks - d_idx_sig_1p
    
    # Filter valid peaks
    peaks = peaks[peaks >= 0]
    n_rev = len(peaks) - 1

    rev_interpolated = []

    # Loop over all revolutions and interpolate to the phase grid
    for j in range(n_rev):
        idx_start = peaks[j]
        idx_end = peaks[j+1]
        
        y_rev = y[idx_start:idx_end+1]
        
        # Current phase vector for this specific revolution
        ph_rev = np.linspace(0, 2 * np.pi, len(y_rev))
        
        # Interpolate to the master phase grid (ph_intp)
        f_interp = interp1d(ph_rev, y_rev, kind='linear', fill_value="extrapolate")
        rev_interpolated.append(f_interp(ph_intp))

    # Convert to array and compute ensemble average
    rev_interpolated = np.array(rev_interpolated)
    y_avg = np.mean(rev_interpolated, axis=0)
    
    return y_avg, rev_interpolated

# ============================================================
# 2. MAIN PROCESSING SCRIPT
# ============================================================
def main():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fn_folder = os.path.join(script_dir, 'Mic')
    
    # Inputs
    fn_list = ['DPN18.txt']
    prop_diameter = 0.4064
    fs = 51200.0  # Sampling frequency
    ph_intp = np.linspace(0, 2 * np.pi, 361)[:-1] # 0 to 360 deg
    d_ph = 0

    results = []

    for fn in fn_list:
        avg_path = os.path.join(fn_folder, fn)
        
        # Load metadata (Assumes space/comma separated numeric file)
        # We skip the columns used in MATLAB and Map them
        avg_data = pd.read_csv(avg_path, header=None).values
        
        dpn_list = avg_data[:, 0].astype(int)
        v_inf = avg_data[:, 6]
        aoa_list = avg_data[:, 12]
        rps_m1 = avg_data[:, 14]

        all_y_avg = []
        all_p_rms = []

        for idx, dpn in enumerate(dpn_list):
            # Construct TDMS filename
            # Format: propOn_..._run[DPN]_001.tdms
            tdms_filename = f"{fn[:-4]}_run{dpn}_001.tdms"
            tdms_path = os.path.join(fn_folder, tdms_filename)

            if not os.path.exists(tdms_path):
                print(f"File not found: {tdms_path}")
                continue

            print(f"Processing: {tdms_filename}")
            
            # Read TDMS using nptdms
            tdms_file = TdmsFile.read(tdms_path)
            # Access the first group (Matlab rawData{1})
            group = tdms_file.groups()[0]
            
            # Extract Microphone (Col 1) and Trigger (Col 2)
            # Note: We access by index. Adjust if your channels are named differently
            p_mic = group.channels()[0].data
            one_p = group.channels()[1].data

            # Calculate Phase Average
            y_avg, rev_matrix = phase_avg_data(p_mic, one_p, fs, rps_m1[idx], ph_intp, d_ph)
            
            all_y_avg.append(y_avg)
            # Calculate RMS of the tonal (averaged) content
            all_p_rms.append(np.sqrt(np.mean(y_avg**2)))

        # Store for plotting
        results.append({
            'aoa': aoa_list,
            'y_avg': np.array(all_y_avg),
            'p_rms': np.array(all_p_rms)
        })

    # ============================================================
    # 3. PLOTTING (Replicating MATLAB Figures)
    # ============================================================
    # Figure 1: Phase-averaged signals for first two data points
    plt.figure(figsize=(10, 8))
    for j in range(min(2, len(results[0]['aoa']))):
        plt.subplot(2, 1, j+1)
        plt.plot(np.rad2deg(ph_intp), results[0]['y_avg'][j])
        plt.title(f"AoA = {results[0]['aoa'][j]:.1f} deg")
        plt.xlabel("Phase angle [deg]")
        plt.ylabel("Acoustic pressure [Pa]")
    plt.tight_layout()

    # Figure 2: RMS trend over AoA
    plt.figure()
    plt.plot(results[0]['aoa'], results[0]['p_rms'], 'bo', label='Tonal Content')
    plt.xlabel("AoA [deg]")
    plt.ylabel("pRMS tonal content [Pa]")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()