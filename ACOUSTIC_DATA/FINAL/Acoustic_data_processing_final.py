import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
from nptdms import TdmsFile
import seaborn as sns
import matplotlib as mpl

# ── Colour palette (match notebook) ───────────────────────────────────────────
sns.set_palette("colorblind")
COLORS = sns.color_palette("colorblind").as_hex()
# ── Global font / style settings (match notebook) ─────────────────────────────
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",

    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 12,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "legend.frameon": True,
})

# ============================================================
# 1. HELPER FUNCTIONS: PERFORMANCE & CONVECTION
# ============================================================

def get_ct(J):
    """
    Returns the Thrust Coefficient (Ct) based on the BEM response model
    provided in the assignment manual.
    Valid for J between 0.8 and 5.0.
    """
    return -0.0051*J**4 + 0.0959*J**3 - 0.5888*J**2 + 1.0065*J - 0.1353

def get_convection_results(X_m, Y_m, Z_m, V_inf, c=343.0):
    """
    Transforms Measured (Tunnel) coordinates to Acoustic (Still-Air) coordinates.
    """
    M = V_inf / c
    beta = np.sqrt(1 - M**2)
    
    r = np.sqrt(X_m**2 + Y_m**2 + Z_m**2)
    theta_prime = np.degrees(np.arccos(-X_m / r))
    
    r_prime = (-M * X_m + np.sqrt(X_m**2 + beta**2 * (Y_m**2 + Z_m**2))) / beta**2
    theta = np.degrees(np.arccos(-(X_m - M * r_prime) / r_prime))
    
    X_new = -r * np.cos(np.deg2rad(theta))
    h_new = r * np.sin(np.deg2rad(theta))
    h_old = np.sqrt(Y_m**2 + Z_m**2)
    Y_new = h_new * (Y_m / h_old)
    Z_new = h_new * (Z_m / h_old)
    
    delta_db = 20 * np.log10(r_prime / r)
    
    return {
        'xyz_mic': [X_m, Y_m, Z_m],
        'xyz_r_prime': [X_new, Y_new, Z_new],
        'theta_prime': theta_prime,
        'theta': theta,
        'r': r,
        'r_prime': r_prime,
        'delta_db': delta_db
    }

# ============================================================
# 2. SPECTRAL PROCESSING (DUAL REFERENCE & OASPL)
# ============================================================

def calculate_spsl(p_mic, fs, p_ref_nd, bpf, N=8192):
    """
    Calculates Absolute SPSL, Fully Non-Dimensional SPSL, and OASPL.
    """
    T_win, data = N / fs, p_mic - np.mean(p_mic)
    B = len(data) // N
    num_segments, hop_size = 2 * B - 1, N // 2 
    phi_accum = np.zeros(N // 2 + 1)
    wj = signal.windows.hann(N, sym=False)
    window_energy = np.sum(wj**2)
    
    for part in range(num_segments):
        start = part * hop_size
        X = np.fft.rfft(wj * data[start : start + N])
        phi_accum += np.abs(X)**2

    phi = (phi_accum / num_segments) * (1 / (N * window_energy)) * T_win * 2.0 
    freqs = np.fft.rfftfreq(N, 1/fs)

    spsl_abs = 10 * np.log10((phi / (20e-6)**2) + 1e-12) 

    spsl_nd = 10 * np.log10((phi * bpf) / (p_ref_nd**2) + 1e-25) 

    df = fs / N 
    
    # OASPL Calculation over the whole frequency spectrum
    p_rms_sq = np.sum(phi) * df

    # 1. Absolute OASPL (20 uPa ref)
    oaspl_abs = 10 * np.log10((p_rms_sq / (20e-6)**2) + 1e-12)
    
    # 2. Non-Dimensional OASPL (T/D^2 ref)
    oaspl_nd = 10 * np.log10((p_rms_sq / (p_ref_nd)**2) + 1e-25)
    
    return freqs, spsl_abs, spsl_nd, oaspl_abs, oaspl_nd

# ============================================================
# 3. MAIN EXECUTION
# ============================================================

def main():
    # Setup coordinates [0.55, 0.44, 0.43] meters
    Xm, Ym, Zm = 0.55, 0.44, 0.43
    fs, D, num_blades = 51200.0, 0.2032, 6
    rho = 1.225
    c_sound = 343.0
    
    target_files = ['DPN18.txt', 'DPN19.txt', 'DPN22.txt', 'DPN26.txt', 'DPN27.txt']
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mic_folder = os.path.join(script_dir, 'Mic')
    output_folder = os.path.join(script_dir, 'Generated plots')
    os.makedirs(output_folder, exist_ok=True)

    results = []
    for fn in target_files:
        path = os.path.join(mic_folder, fn)
        if not os.path.exists(path): continue
        meta = pd.read_csv(path, sep=',', header=None)
        
        for _, row in meta.iterrows():
            dpn, v_inf, aoa, rps = int(float(row[0])), float(row[6]), round(float(row[12]), 1), float(row[14])
            tdms_p = os.path.join(mic_folder, f"{fn[:-4]}_run{dpn}_001.tdms")
            
            if os.path.exists(tdms_p):
                J = v_inf / (rps * D)
                ct = get_ct(J)
                thrust_isolated = ct * rho * (rps**2) * (D**4)
                p_ref_thrust = abs(thrust_isolated) / (D**2)
                
                bpf = rps * num_blades
                mach = v_inf / c_sound
                
                p_raw = TdmsFile.read(tdms_p).groups()[0].channels()[0].data
                
                # Unpack both OASPL variables
                freqs, spsl_abs, spsl_nd, oaspl_abs_raw, oaspl_nd_raw = calculate_spsl(p_raw, fs, p_ref_thrust, bpf)
                c = get_convection_results(Xm, Ym, Zm, v_inf, c=c_sound)
                
                results.append({
                    'dpn': dpn, 'j': round(J, 2), 'aoa': aoa, 'rps': rps,
                    'mach': mach,
                    'freqs': freqs, 
                    'spsl_raw_abs': spsl_abs, # Store the uncorrected absolute SPSL
                    'spsl_corr_abs': spsl_abs + c['delta_db'],
                    'spsl_corr_nd': spsl_nd + c['delta_db'],
                    'oaspl_corr_abs': oaspl_abs_raw + c['delta_db'],
                    'oaspl_corr_nd': oaspl_nd_raw + c['delta_db'],
                    'p_ref': p_ref_thrust,
                    'ct': ct,
                    **c 
                })
    
    df = pd.DataFrame(results)

    # --- TABLE 1: AEROACOUSTIC & OASPL SUMMARY ---
    print("\n" + "═"*75)
    print(f"{'DUAL-SCALED AEROACOUSTIC & OASPL SUMMARY':^75}")
    print("═"*75)
    if not df.empty:
        print(f"{'DPN':<6} | {'J':<5} | {'AoA':<5} | {'Ct':<8} | {'OASPL (Abs)':<12} | {'OASPL (ND)':<12}")
        print("-" * 75)
        for _, r in df.iterrows():
            print(f"{r['dpn']:<6} | {r['j']:<5.2f} | {r['aoa']:<5.1f} | {r['ct']:<8.4f} | {r['oaspl_corr_abs']:>7.2f} dB  | {r['oaspl_corr_nd']:>7.2f} dB")
    print("═"*75 + "\n")

    # --- TABLE 2: CONVECTION CORRECTION SUMMARY ---
    print("\n" + "═"*82)
    print(f"{'CONVECTION CORRECTION SUMMARY':^82}")
    print("═"*82)
    if not df.empty:
        print(f"{'DPN':<6} | {'Mach':<6} | {'r [m]':<7} | {'theta_p [°]':<11} | {'theta [°]':<9} | {'r_prime [m]':<11} | {'Diff [dB]':<10}")
        print("-" * 82)
        for _, r in df.iterrows():
            # Difference: Uncorrected - Corrected = -delta_db
            diff_db = -r['delta_db']
            print(f"{r['dpn']:<6} | {r['mach']:<6.4f} | {r['r']:<7.4f} | {r['theta_prime']:<11.2f} | {r['theta']:<9.2f} | {r['r_prime']:<11.4f} | {diff_db:>6.2f} dB")
    print("═"*82 + "\n")

    # =========================================================================
    # --- PLOT 1: J Sweep - ABSOLUTE FREQUENCY (20uPa ref, Constant AoA) ---
    # =========================================================================
    plt.figure(figsize=(7, 4))
    j_data = df[df['aoa'] == 2.5].sort_values('j')
    
    j_colors = {1.6: COLORS[0], 2.0: COLORS[1], 2.8: COLORS[2]}

    for i, (_, row) in enumerate(j_data.iterrows()):
        current_j = round(row['j'], 1)
        color = j_colors.get(current_j, 'black') 
        
        bpf = row['rps'] * num_blades
        
        plt.plot(row['freqs'], row['spsl_corr_abs'], color=color,
                 label=f"J = {row['j']:.1f}", linewidth=1.2)
        
        for n in range(1, 5):
            harmonic_freq = n * bpf
            line_label = f"BPF (J={current_j}, 1st BPF ≈ {bpf:.0f}Hz)" if i < len(j_colors) and n == 1 else None
            
            plt.axvline(x=harmonic_freq, 
                        color=color, 
                        linestyle='--', 
                        alpha=0.7, 
                        linewidth=0.9,
                        label=line_label)
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"SPSL [dB/Hz] ($p_{ref}=20 \mu Pa$)")
    plt.xlim([11, 3600]) 
    plt.ylim([35, 85])
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.8)
    plt.savefig(os.path.join(output_folder, 'J_Sweep_Absolute_AbsFreq.pdf'), dpi=300, bbox_inches='tight')

    # =========================================================================
    # --- PLOT 2: J-SWEEP (FULLY ND SCALED) ---
    # =========================================================================
    plt.figure(figsize=(7, 4))
    for i, (_, row) in enumerate(j_data.iterrows()):
        bpf = row['rps'] * num_blades
        normalized_freqs = row['freqs'] / bpf
        plt.plot(normalized_freqs, row['spsl_corr_nd'], label=f"J = {row['j']}", linewidth=1.2)
        
    plt.xlabel(r"Normalized frequency ($f/BPF$)")
    plt.ylabel(r"SPSL [dB] ($p_{ref}=T/D^2$)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlim([10/421, 4.2])
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.8)

    plt.savefig(os.path.join(output_folder, 'J_Sweep_Thrust_BPF_Normalized.pdf'), dpi=300, bbox_inches='tight')

    # =========================================================================
    # --- PLOT 3: ALPHA-SWEEP (FULLY ND SCALED) ---
    # =========================================================================
    plt.figure(figsize=(7, 4))
    aoa_data = df[abs(df['j'] - 1.6) <= 0.05].sort_values('aoa')
    for _, row in aoa_data.iterrows():
        bpf = row['rps']*num_blades
        plt.plot(row['freqs']/bpf, row['spsl_corr_nd'], label=f"$\\alpha$ = {row['aoa']}°", linewidth=1.2)
   
    plt.xlabel("Normalized frequency ($f/BPF$)")
    plt.ylabel(r"SPSL [dB] ($p_{ref}=T/D^2$)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlim([10/421, 4.2])
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.8)

    plt.savefig(os.path.join(output_folder, 'Alpha_Sweep_ThrustScaled.pdf'), dpi=300, bbox_inches='tight')

    # =========================================================================
    # --- PLOT 4: UNCORRECTED VS CORRECTED SPSL (Absolute Freq, 20uPa) ---
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    # Plotting this comparison for the first valid run in the dataframe (e.g., J=1.6, AoA=2.5)
    sample_row = df.iloc[0]
    
    plt.plot(sample_row['freqs'], sample_row['spsl_raw_abs'], 
             label='Uncorrected', color=COLORS[0], linewidth=1.2)
    plt.plot(sample_row['freqs'], sample_row['spsl_corr_abs'], 
             label='Corrected', color=COLORS[1], linewidth=1.2)
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"SPSL [dB/Hz] ($p_{ref}=20 \mu Pa$)")
    plt.xlim([11, 3600]) 
    plt.ylim([45, 85])
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.8)
    
    plt.savefig(os.path.join(output_folder, 'Uncorrected_vs_Corrected_SPSL.pdf'), dpi=400, bbox_inches='tight')

    plt.show()

if __name__ == "__main__": main()