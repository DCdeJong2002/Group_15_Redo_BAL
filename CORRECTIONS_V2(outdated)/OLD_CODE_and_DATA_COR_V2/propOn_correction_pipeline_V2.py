from pathlib import Path
import pandas as pd

from CORRECTIONS_V2.OLD_CODE_and_DATA_COR_V2.correction_classes import ModelOffCorrector, PropOffData, PropOnData


BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Example 1: prop-on workflow
# raw prop-on -> model-off -> solid blockage -> wake blockage
# ------------------------------------------------------------

prop_on_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "all_rudder_cases_combined.csv")

modeloff = ModelOffCorrector(
    correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
    save_dir=BASE_DIR / "results_propOn"
)

df_prop_on_modeloff = modeloff.apply(
    prop_on_raw,
    save_csv=True,
    filename="propOn_modeloff_corrected.csv"
)

propon = PropOnData(df_prop_on_modeloff, velocity_tolerance=1.0)
propon.set_save_directory(BASE_DIR / "results_propOn")

df_prop_on_solid = propon.apply_solid_blockage(
    save_csv=True,
    filename="propOn_solid_blockage_corrected.csv"
)

df_prop_on_wake = propon.apply_wake_blockage(
    fit_csv=BASE_DIR / "results_propOff" / "propOff_CD_fit_parameters.csv",
    save_csv=True,
    filename="propOn_wake_blockage_corrected.csv"
)




