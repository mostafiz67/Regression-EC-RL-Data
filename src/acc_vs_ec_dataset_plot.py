"""
Some code is inherited from https://stackoverflow.com/questions/68123724/how-to-plot-multiple-csv-files-with-separate-plots-for-each-category
"""

# python3 -m src.acc_vs_ec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import OUT, PLOT_OUTPUT_PATH


CSVS = [
    OUT / "Wine_error.csv",
    OUT / "Bike_error.csv",
    # OUT / "House_error.csv",
    # OUT / 'Breast-Cancer_error.csv',
    # OUT / 'Cancer_error.csv',
    # OUT / 'Diabetics_error.csv',
    # OUT / 'Parkinsons-Tele_error.csv',
]

df = pd.concat([pd.read_csv(csv).assign(Dataset=csv.stem) for csv in CSVS]).reset_index(drop=True)
df = df.drop(df[(df.Regressor == "SVR") | (df.Regressor == "Knn") | (df.Regressor == "RF") |
                (df.Regressor == "Ridge") | (df.Regressor == "Lasso") | 
                (df.Method == "negative_incon") | (df.Method == "positive_incon")].index) # removing non hypertune models data

df['Dataset'].replace(regex=True,inplace=True,to_replace='_error',value=r'')
df['Regressor'].replace(regex=True,inplace=True,to_replace='-H',value=r'')

def ec_vs_accuracy_dataset():
    for dataset in df.Dataset.unique():
        print(df, dataset)
        data = df[df["Dataset"] == dataset]
        print(data)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=data, x="EC", y="MAE", hue="Method", style="Regressor", s=100)
        plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0, loc="center left")
        plt.xlabel("Error Consistency (EC)")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title(f"Error Consistency vs MAE for the {dataset} dataset ")
        plt.savefig(PLOT_OUTPUT_PATH / 'new_medical_plots' / f"MAE_{dataset}_only.png", bbox_inches='tight')
        plt.clf()


def ec_vs_accuracy_all_dataset():
    # print(df, dataset)
    data = df
    print(data)
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(data=data, x="EC", y="MAE", hue="Method", style="Regressor", s=100)
    main = sns.catplot(data=data, x="EC", y="MAE", hue="Dataset", col="Method", kind="point", col_wrap=2, height=10, aspect=3,
    sharex=False, sharey=False)
    # plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0, loc="center left")
    # axes = main.axes
    # axes[0,0].set_ylim(0, )
    # axes[0, 1].set_ylim(0, )
    # main.set(ylim(0, None))
    plt.xlabel("Error Consistency (EC)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Error Consistency vs MAE for alldataset ")
    plt.savefig(PLOT_OUTPUT_PATH / 'new_medical_plots' / f"MAE_all_dataset.png", bbox_inches='tight')
    plt.clf()



if __name__ == "__main__":
    # ec_vs_accuracy_dataset()
    ec_vs_accuracy_all_dataset()