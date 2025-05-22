import pandas as pd
import numpy as np
import os  # ← (added import that seemed to be implicitly required)

# SET WORKING PATH
working_directory = (
    r"C:\Users\s.parisi\OneDrive - diagramgroup.it\STELAR\RICOSTRUISCI_DATI_MANCANTI"
)
os.chdir(working_directory)

# INPUT DATA
dati_df = pd.read_excel(
    "DATI_NETSENS_TORREVILLA.xlsx"
)  # WEATHER DATA TO BE RECONSTRUCTED
coords_df = pd.read_excel("COORD.xlsx")  # WEATHER-STATION COORDINATES

# OUTPUT FILE
output_filename = "DATI_NETSENS_TORREVILLA_INTERPOLATI.xlsx"

##########################################


# Function to compute the Euclidean distance between two points (lat, lon)
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Euclidean distance between two points — (lat1, lon1) and (lat2, lon2).
    """
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)


# IDW function to interpolate missing values
def idw_interpolation(dati_df, coords_df, stazione, target_date, variabile, alpha=2):
    """
    Perform IDW interpolation for a given date and station, using the other stations.
    """
    print(
        f"Interpolating for station: {stazione}, variable: {variabile}, date: {target_date}"
    )

    # Column name that combines station and variable (e.g. 'BARB_TMED')
    column_name = f"{stazione}_{variabile}"

    # Extract the row(s) for the specified date
    station_data = dati_df[dati_df["DATE"] == target_date]

    # If the value for the station is missing …
    if pd.isna(station_data[column_name].values[0]):
        print(
            f"Missing data for station {stazione} on {target_date}. Performing interpolation..."
        )

        # Coordinates of the target station
        stazione_coords = coords_df[coords_df["STAZ"] == stazione]
        lat_stazione, lon_stazione = (
            stazione_coords["LAT"].values[0],
            stazione_coords["LON"].values[0],
        )

        # Gather data from the other stations (exclude the target one)
        available_data = []
        distances = []

        for other_stazione in ["BARB", "SCHI", "MOND", "TORR"]:
            if other_stazione != stazione:  # Skip the station being interpolated
                col_name = f"{other_stazione}_{variabile}"
                other_coords = coords_df[coords_df["STAZ"] == other_stazione]
                lat_other, lon_other = (
                    other_coords["LAT"].values[0],
                    other_coords["LON"].values[0],
                )

                # Compute distance between the two stations
                distance = calculate_distance(
                    lat_stazione, lon_stazione, lat_other, lon_other
                )

                # Store the value and its distance
                available_data.append(station_data[col_name].values[0])
                distances.append(distance)

        # Inverse-distance weights
        weights = [1 / (d**alpha) if d != 0 else 1 for d in distances]

        # Weighted average (if total weight ≠ 0)
        if sum(weights) > 0:
            interpolated_value = np.sum(np.multiply(weights, available_data)) / np.sum(
                weights
            )
            return interpolated_value
        else:
            return np.nan
    else:
        return station_data[column_name].values[0]  # Return the existing value


# Variables to process (TMED, TMAX, TMIN, RR)
variabili = ["TMED", "TMAX", "TMIN", "RR"]

# Process each column for every variable
for variabile in variabili:
    for col in dati_df.columns:
        if variabile in col:  # Only columns that match the variable
            # Extract station and variable from the column name
            stazione, _ = col.split("_")
            print(
                f"Processing column: {col}  |  station: {stazione}  |  variable: {variabile}"
            )

            # Interpolate missing values in-place
            dati_df[col] = dati_df.apply(
                lambda row: (
                    idw_interpolation(
                        dati_df, coords_df, stazione, row["DATE"], variabile
                    )
                    if pd.isna(row[col])
                    else row[col]
                ),
                axis=1,
            )

# Save the result to a new Excel file
dati_df.to_excel(output_filename, index=False)

# Confirmation message
print(f"Interpolated data have been saved to: {output_filename}")
