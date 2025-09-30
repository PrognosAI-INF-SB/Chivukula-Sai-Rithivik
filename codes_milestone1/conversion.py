import os
import pandas as pd

'''input_folder = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/train_and_test"

output_folder = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/raw data"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define column names based on NASA CMAPSS dataset specifications
column_names = [
    "unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
    "sensor_7", "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", "sensor_18",
    "sensor_19", "sensor_20", "sensor_21"
]

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    # Process only text files
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)

        # Load the text file into a pandas DataFrame
        # The data is space-separated and has no header row
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=column_names)

        # Remove any completely empty columns (if present)
        df = df.dropna(axis=1, how="all")

        # Create the CSV file name by replacing .txt with .csv
        csv_name = file_name.replace(".txt", ".csv")
        output_path = os.path.join(output_folder, csv_name)

        # Save the DataFrame as a CSV file without the index
        df.to_csv(output_path, index=False)

        # Print confirmation
        print(f"Converted {file_name} -> {csv_name}")
 '''



# Folder containing the RUL text files
input_folder = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/rul"

# Folder to save CSVs
output_folder = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/raw data"
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the folder
for file_name in os.listdir(input_folder):
    if "RUL" in file_name and file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)
        
        # Read the RUL file (single column)
        rul_df = pd.read_csv(file_path, sep="\s+", header=None, names=["RUL"])
        
        # Save as CSV
        csv_name = file_name.replace(".txt", ".csv")
        rul_df.to_csv(os.path.join(output_folder, csv_name), index=False)
        
        print(f"Converted {file_name} -> {csv_name}")

print("All RUL files converted and saved in:", output_folder)
