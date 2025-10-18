import pandas as pd
# The 'io' library is no longer needed.

# Filepath for the dataset
filepath = 'uci_raw.txt'

# 1. Load the dataset from the text file
# We parse the 'Date' and 'Time' columns into a single datetime index called 'dt'.
# Missing values are specified as '?' or 'nan'.
df = pd.read_csv(
    filepath,
    sep=';',
    parse_dates={'dt' : ['Date', 'Time']},
    infer_datetime_format=True,
    low_memory=False,
    na_values=['nan', '?'],
    index_col='dt'
)

# 2. Handle potential missing values
# For this demonstration, we'll fill missing values with the value from the previous minute.
df.ffill(inplace=True)

# 3. Convert Global_active_power to Watt-Hours per minute
# Global_active_power is in kilowatts; this step copies it as-is (no conversion yet).
df['watt_hour_consumption'] = (df['Global_active_power'])

# 4. Resample to get mean hourly power
# We take the mean of the power readings for each hour.
hourly_consumption = df['watt_hour_consumption'].resample('H').mean()

# 5. Create the final DataFrame
# It contains just the timestamp and the calculated hourly consumption.
final_df = pd.DataFrame({
    'hourly_consumption': hourly_consumption
})
final_df.index.name = 'timestamp'

# 6. Save the transformed data to a new CSV file
output_filename = 'uci_dataset.csv'
final_df.to_csv(output_filename)
print(f"Transformed data saved to '{output_filename}'")

# 7. Load the data back to demonstrate it's ready for use
# This step confirms the file is saved correctly and can be loaded for analysis.
loaded_df = pd.read_csv(output_filename, index_col='timestamp', parse_dates=True)

print("\n--- Transformed Data ---")
print("The transformed and resampled data is ready for time series analysis.")
print("The first 5 rows of the loaded file are:")
print(loaded_df.head(48))
