import pandas as pd

# file paths
excel_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\AERODYNAMIC_DATA_propoff\propOff.xlsx"
csv_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\AERODYNAMIC_DATA_propoff\propOff.csv"

# read excel
df = pd.read_excel(excel_file)

# save as csv
df.to_csv(csv_file, index=False)

print("Conversion complete.")