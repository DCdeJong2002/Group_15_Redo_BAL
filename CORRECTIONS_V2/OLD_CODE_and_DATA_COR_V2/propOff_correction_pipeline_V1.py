from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, PropOffData, PropOnData


BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Example 2: prop-off workflow
# raw prop-off -> optional model-off -> solid blockage -> fit -> wake blockage
# ------------------------------------------------------------

prop_off_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "propOff.csv")

propoff = PropOffData(prop_off_raw)
propoff.set_save_directory(BASE_DIR / "results_propOff_noModelOff")

df_prop_off_solid = propoff.apply_solid_blockage(
    save_csv=True,
    filename="propOff_solid_blockage_corrected.csv"
)

df_prop_off_fit = propoff.fit_cd_polar(
    save_csv=True,
    filename="propOff_with_CD_fit_values.csv",
    fit_params_filename="propOff_CD_fit_parameters.csv"
)

df_prop_off_wake = propoff.apply_wake_blockage(
    save_csv=True,
    filename="propOff_wake_blockage_corrected.csv"
)

