# AE4115 Wind Tunnel Data Correction Pipeline

## Quick Start — Run in This Order

```bash
# 1. Parse raw balance files into processed CSVs
python 01_raw_data_correction/correct_input_raw_files.py
python 01_raw_data_correction/TAILOFF_correct_input_raw_files.py

# 2. (Once only) Generate the model-off correction grid
python 02p2_processed_data_corrections/generate_model_off_data_FINAL.py

# 3. Run aerodynamic correction pipelines (from inside 02p2_processed_data_corrections/)
cd 02p2_processed_data_corrections
python propOff_correction_pipeline_FINAL.py
python propOn_correction_pipeline_FINAL.py   # depends on prop-off results
python TAILOFF_correction_pipeline_FINAL.pypython 

# 4. Open analysis notebooks
03_corrected_data_analysis/correction_comparison_FINAL_clean.ipynb
03_corrected_data_analysis/data_plotter_FINAL_clean.ipynb

# Run acoustic processing is independent of balance data
python 04_acoustic_data_processing/Acoustic_data_processing_final.py


```

---

## Folder Structure

```
├── 01_raw_data_correction/          # Raw .txt balance files → processed CSVs
│   ├── RAW_TEST_DATA/               # Raw prop-on/off balance measurements
│   ├── RAW_TAILOFF_DATA/            # Raw tail-off balance measurements
│   ├── PROCESSED_TEST_DATA/         # Output: rudder sweep CSVs
│   └── PROCESSED_TAILOFF_DATA/      # Output: tail-off combined CSV
│
├── 02p1_analytical_corrections/     # Manual correction coefficients (Excel + plots)
│   ├── CORRECTIONS_SHEET.xlsx       # τ₁, τ₂, K, δ, bv lookup values
│   └── DIGITIZED_PLOTS/             # Scripts to digitize Barlow & Pope figures
│
├── 02p2_processed_data_corrections/ # Main correction pipeline
│   ├── correction_classes_FINAL.py  # All correction logic (ModelOff, PropOff, PropOn, TailOff)
│   ├── propOff_correction_pipeline_FINAL.py
│   ├── propOn_correction_pipeline_FINAL.py
│   ├── TAILOFF_correction_pipeline_FINAL.py
│   ├── generate_model_off_data_FINAL.py
│   ├── INPUT_BALANCE_DATA/          # Input: propOff.csv, CT experimental data
│   ├── MODEL_OFF_DATA/              # model_off_corrections_grid.csv (generated)
│   ├── results_propOff_FINAL/       # Output: propOff_final.csv
│   ├── results_propOn_FINAL/        # Output: propOn_final.csv, _BEM.csv, _EXP.csv
│   └── results_TAILOFF_FINAL/       # Output: TAILOFF correction stage CSVs
│
├── 03_corrected_data_analysis/      # Jupyter notebooks for plotting & comparison
│   ├── correction_comparison_FINAL_clean.ipynb
│   └── data_plotter_FINAL_clean.ipynb
│
└── 04_acoustic_data_processing/     # Microphone TDMS processing & spectral plots
    ├── Acoustic_data_processing_final.py
    ├── Mic/                         # Input: DPN*.txt metadata + .tdms data files
    └── Generated plots/             # Output: spectral plots (auto-created)
```

---

## Correction Stages (02p2)

Each pipeline applies corrections sequentially, tracked via an `active_cols` dictionary:

0. **Thrust correction** *(prop-on only)*— subtracts thrust using EXP data
1. **Model-off tare** — subtracts wind-on support interference
2. **Solid blockage** — corrects for model volume in tunnel cross-section
3. **Wake blockage** — Maskell's method for separated wake effects
4. **Slipstream blockage** *(prop-on only)* — additional blockage from propeller slipstream
5. **Streamline curvature** — corrects CM_pitch and CL for curved streamlines
6. **Downwash** — tail-off CL–α slope used to correct horizontal tail contribution
7. **Tail correction** — removes HTP aerodynamic contribution using tail-off data

Toggle any stage via boolean flags in each pipeline's `run_*_workflow()` call.

---

## Dependencies

```
numpy  pandas  scipy  matplotlib  seaborn  openpyxl  nptdms
```

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl nptdms
```
