import pandas as pd

def convert_csv_format(input_file='ha_raw.csv', output_file='ha_converted.csv'):
    """
    Loads a CSV file in format A, converts it to format B, and saves it.

    Format A:
    Time,HAData_Consumption
    1680822000000,0.18

    Format B:
    timestamp,hourly_consumption
    2023-04-06 23:00:00,0.18

    Args:
        input_file (str): The path to the input CSV file (Format A).
        output_file (str): The path to save the converted CSV file (Format B).
    """
    try:
        # Read the source CSV file
        df = pd.read_csv(input_file)

        # Ensure the column names are as expected, stripping any whitespace
        df.columns = df.columns.str.strip()

        # Convert the 'Time' column from Unix milliseconds to datetime objects
        # The unit 'ms' specifies that the source values are in milliseconds
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')

        # Rename the columns to match the target format
        df.rename(columns={
            'Time': 'timestamp',
            'HAData_Consumption': 'hourly_consumption'
        }, inplace=True)

        # Save the transformed dataframe to the new CSV file
        # index=False prevents pandas from writing the dataframe index as a column
        df.to_csv(output_file, index=False)

        print(f"Successfully converted '{input_file}' and saved it as '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # This allows the script to be run directly from the command line
    convert_csv_format()
