import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

from src.constants import OUT, PLOT_OUTPUT_PATH


CSVS = [
    OUT / "Wine_error.csv",
    OUT / "Bike_error.csv",
    OUT / "House_error.csv",
    OUT / 'Breast-Cancer_error.csv',
    OUT / 'Cancer_error.csv',
    OUT / 'Diabetics_error.csv',
    OUT / 'Parkinsons-Tele_error.csv',
]

df = pd.concat([pd.read_csv(csv).assign(Dataset=csv.stem) for csv in CSVS]).reset_index(drop=True)


def ec_vs_accuracy():
    for method in df.Method.unique():
        data = df[df.Method.eq(method)]
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=data, x="EC", y="R2", hue="Dataset", style="Regressor", s=100)
        plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0, loc="center left")
        plt.xlabel("Error Consistency (EC)")
        plt.ylabel("R-Squared (R2)")
        plt.title(f"Error Consistency ({method}) vs R2")
        plt.savefig(PLOT_OUTPUT_PATH / 'new_medical_plots' / f"{method}_R2.png", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    ec_vs_accuracy()