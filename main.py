import json
import sys
import traceback
import os
import tempfile
import pandas as pd
import numpy as np
import time
from utils.mclient import MinioClient
from utils.metrics import calc_mae, calc_mre, calc_rmse, calc_mse
from sklearn.metrics import r2_score

def compute_metrics(original_df: pd.DataFrame, missing_df: pd.DataFrame, imputed_df: pd.DataFrame) -> dict:
    """
    Scale back the data to the original representation.
    :param original_df: original/ground truth dataframe.
    :param missing_df:  dataframe with the missing values.
    :param imputed_df: dataframe with the imputed values.
    :return: dictionary with the calculated metrics
    """
    mask = np.isnan(missing_df) ^ np.isnan(original_df)
 
    metrics = dict()
    original_df_np = np.nan_to_num(original_df.to_numpy())
    imputed_df_np = imputed_df.to_numpy()
    mask_np = mask.to_numpy()
 
    metrics['Missing value percentage'] = (missing_df.isna().sum().sum() * 100) / (
            missing_df.shape[0] * missing_df.shape[1])
 
    metrics['Mean absolute error'] = calc_mae(
        imputed_df_np,
        original_df_np,
        mask_np
    )
 
    metrics['Mean square error'] = calc_mse(
        imputed_df_np,
        original_df_np,
        mask_np
    )
 
    metrics['Root mean square error'] = calc_rmse(
        imputed_df_np,
        original_df_np,
        mask_np
    )
 
    metrics['Mean relative error'] = calc_mre(
        imputed_df_np,
        original_df_np,
        mask_np
    )
 
    metrics['Euclidean Distance'] = np.linalg.norm((imputed_df_np - original_df_np)[mask_np])
 
    # Flatten the arrays and apply the mask
    imputed_df_np_flat = imputed_df_np[mask_np].flatten()
    original_df_np_flat = original_df_np[mask_np].flatten()
 
    # Calculate the R-squared score
    metrics['r2 score'] = r2_score(original_df_np_flat, imputed_df_np_flat)
 
    return metrics


def run(json_blob):
    """
    Entrypoint for the STELAR tool.  Downloads the two Excel inputs from MinIO,
    performs the IDW interpolation exactly as in the original script, uploads the
    resulting Excel file back to MinIO, and returns a task-result JSON.
    """

    try:
        # ──────── 1.  MinIO client initialisation ───────────────────────────────
        minio = json_blob["minio"]
        mc = MinioClient(
            minio["endpoint_url"],
            minio["id"],
            minio["key"],
            secure=True,
            session_token=minio["skey"],
        )

        # ──────── 2.  Input / output paths from task JSON ───────────────────────
        # Expected keys:
        #   json_blob["inputs"]["dati_file"]      – path to METEO data Excel
        #   json_blob["inputs"]["coords_file"]    – path to station-coords Excel
        #   json_blob["outputs"]["interpolated_file"] – where to upload the result
        in_dati_obj   = json_blob["input"]["meteo_file"][0]
        in_coords_obj = json_blob["input"]["coords_file"][0]
        out_obj       = json_blob["output"]["interpolated_file"]

        if "ground_truth" in json_blob["input"]:
            gt_obj = json_blob["input"]["ground_truth"][0]

        # ──────── 3.  Work inside a temporary directory ─────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            dati_local   = os.path.join(tmpdir, "dati.xlsx")
            coords_local = os.path.join(tmpdir, "coords.xlsx")
            out_local    = os.path.join(tmpdir, "interpolated.xlsx")
            gt_local     = os.path.join(tmpdir, "ground_truth.xlsx")

            mc.get_object(s3_path=in_dati_obj, local_path=dati_local)
            mc.get_object(s3_path=in_coords_obj, local_path=coords_local)

            # ──────── 4.  Original data-analysis code (UNMODIFIED) ──────────────

            # INPUT DATA
            dati_df   = pd.read_excel(dati_local)    # WEATHER DATA TO BE RECONSTRUCTED
            missing_df = pd.read_excel(dati_local)  # DataFrame with missing values
            coords_df = pd.read_excel(coords_local)  # WEATHER-STATION COORDINATES

            # Function to compute Euclidean distance between two points (lat, lon)
            def calculate_distance(lat1, lon1, lat2, lon2):
                return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

            # IDW interpolation
            def idw_interpolation(dati_df, coords_df, stazione, target_date, variabile, alpha=2):
                print(f"Interpolating for station: {stazione}, variable: {variabile}, date: {target_date}")
                column_name = f"{stazione}_{variabile}"
                station_data = dati_df[dati_df["DATE"] == target_date]

                if pd.isna(station_data[column_name].values[0]):
                    print(f"Missing data for station {stazione} on {target_date}. Performing interpolation …")

                    stazione_coords = coords_df[coords_df["STAZ"] == stazione]
                    lat_stazione, lon_stazione = stazione_coords["LAT"].values[0], stazione_coords["LON"].values[0]

                    available_data, distances = [], []

                    for other_stazione in ["BARB", "SCHI", "MOND", "TORR"]:
                        if other_stazione != stazione:
                            col_name = f"{other_stazione}_{variabile}"
                            other_coords = coords_df[coords_df["STAZ"] == other_stazione]
                            lat_other, lon_other = other_coords["LAT"].values[0], other_coords["LON"].values[0]

                            distance = calculate_distance(lat_stazione, lon_stazione, lat_other, lon_other)
                            available_data.append(station_data[col_name].values[0])
                            distances.append(distance)

                    weights = [1 / (d ** alpha) if d != 0 else 1 for d in distances]

                    if sum(weights) > 0:
                        interpolated_value = np.sum(np.multiply(weights, available_data)) / np.sum(weights)
                        return interpolated_value
                    return np.nan
                else:
                    return station_data[column_name].values[0]

            variabili = ["TMED", "TMAX", "TMIN", "RR"]

            start_time = time.time()
            for variabile in variabili:
                for col in dati_df.columns:
                    if variabile in col:
                        stazione, _ = col.split("_")
                        print(f"Processing column: {col}  |  station: {stazione}  |  variable: {variabile}")
                        dati_df[col] = dati_df.apply(
                            lambda row: idw_interpolation(dati_df, coords_df, stazione, row["DATE"], variabile)
                            if pd.isna(row[col])
                            else row[col],
                            axis=1,
                        )
            end_time = time.time()
            dati_df.to_excel(out_local, index=False)
            print(f"Interpolated data have been saved locally to: {out_local}")

            # ──────── 5.  Upload result back to MinIO ────────────────────────────
            mc.put_object(file_path=out_local, s3_path=out_obj)


            if "ground_truth" in json_blob["input"]:
                mc.get_object(s3_path=gt_obj, local_path=gt_local)

                # Load ground truth data
                gt_df = pd.read_excel(gt_local)

                # Remove the DATE column from all DataFrames
                gt_df = gt_df.drop(columns=["DATE"])
                missing_df = missing_df.drop(columns=["DATE"])
                dati_df = dati_df.drop(columns=["DATE"])

                metrics = compute_metrics(gt_df, missing_df, dati_df)


        # ──────── 6.  Return success payload ────────────────────────────────────
        return {
            "message": "Tool executed successfully!",
            "output": {
                "interpolated_file": out_obj,
            },
            "metrics": metrics if "ground_truth" in json_blob["input"] else {},
            "status": "success",
        }

    # ─────────── Error handling ──────────────────────────────────────────────────
    except Exception:
        print(traceback.format_exc())
        return {
            "message": "An error occurred during data processing.",
            "error": traceback.format_exc(),
            "status": 500,
        }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))