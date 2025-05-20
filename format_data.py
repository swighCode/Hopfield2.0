import pandas as pd
import csv
from collections import defaultdict

def format_data():
    # Read the full dataset
    with open("table_results.csv", "r", newline='') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    # Grouping containers
    gaussian5_rows = []
    gaussian20_rows = []
    gaussian50_rows = []
    mosaic5_rows = []
    mosaic20_rows = []
    mosaic50_rows = []
    eyes_censor_rows = []
    original_modern_rows = []
    original_classical_rows = []

    # Classify rows
    for row in rows:
        method = row["method"]
        model_type = row["model_type"]
        
        if method == ("gaussian_5"):
            gaussian5_rows.append(row)
        elif method == "gaussian_20":
            gaussian20_rows.append(row)
        elif method == "gaussian_50":
            gaussian50_rows.append(row)
        elif method == "mosaic_5":
            mosaic5_rows.append(row)
        elif method == "mosaic_20":
            mosaic20_rows.append(row)
        elif method == "mosaic_50":
            mosaic50_rows.append(row)
        elif method == "eyes_censor":
            eyes_censor_rows.append(row)
        elif method == "original":
            if model_type == "modern":
                original_modern_rows.append(row)
            else:
                original_classical_rows.append(row)

    # Helper function to write to CSV
    def write_csv(filename, rows):
        if rows:
            with open(filename, "w", newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    # Write each category to its own file
    write_csv("table_gaussian5.csv", gaussian5_rows)
    write_csv("table_gaussian20.csv", gaussian20_rows)
    write_csv("table_gaussian50.csv", gaussian50_rows)
    write_csv("table_mosaic5.csv", mosaic5_rows)
    write_csv("table_mosaic20.csv", mosaic20_rows)
    write_csv("table_mosaic50.csv", mosaic50_rows)
    write_csv("table_eyes_censor.csv", eyes_censor_rows)
    write_csv("table_original_modern.csv", original_modern_rows)
    write_csv("table_original_classical.csv", original_classical_rows)


def format_table():

    # 1) Load the long-form data
    df = pd.read_csv("table_results.csv")

    # 2) Pivot: index is both method and model_type, columns are num_images
    wide = df.pivot_table(
        index=["method", "model_type"],
        columns="num_images",
        values="accuracy",
        aggfunc="first"    # there should now be exactly one entry per pair
    )

    # 3) Sort the numeric columns
    wide = wide.reindex(sorted(wide.columns), axis=1)

    # 4) Build a single-row-label by combining method & model_type
    def label(row):
        method, mtype = row
        # Base names exactly as in your example:
        base = {
            "original":    "Original",
            "eyes_censor": "Eyes censored",
            "gaussian_5":  "Gaussian σ=5",
            "gaussian_20": "Gaussian σ=20",
            "gaussian_50": "Gaussian σ=50",
            "mosaic_5":    "Mosaic size=5",
            "mosaic_20":   "Mosaic size=20",
            "mosaic_50":   "Mosaic size=50",
        }[method]
        # Append "(classical)" when needed
        if mtype == "classical":
            base += " (classical)"
        return base

    wide.index = [label(idx) for idx in wide.index]

    # 5) (Optional) If you only want the specific face-counts from your example:
    desired = [1,2,3,4,5,10,15,20,30,40,50,100,140,183]
    wide = wide.reindex(columns=desired)

    # 6) Reset index and write out
    out = wide.reset_index().rename(columns={"index": "Number of Faces"})
    out.to_csv("pivot_table.csv", index=False, float_format="%.16g")


if __name__ == "__main__":
    # format_data()
    format_table()