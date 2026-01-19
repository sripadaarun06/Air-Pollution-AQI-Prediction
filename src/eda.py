import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
print("EDA FILE LOADED")
CSV_PATH = Path("data") / "city_day.csv"
def run_eda(df, target_col):
    print("Running EDA function...")
    print(df.describe())
    plt.figure()
    sns.heatmap(df.select_dtypes(include="number").corr(),cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("output/correlation_heatmap.png")
    plt.close()
if __name__ == "__main__":
    print("MAIN STARTED")
    df = pd.read_csv(CSV_PATH)
    print("Data loaded:", df.shape)
    run_eda(df, "AQI")
    print("EDA FINISHED")
    print(df.columns)