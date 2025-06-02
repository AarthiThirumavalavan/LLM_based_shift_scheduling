import pandas as pd

employee_name_list = []

def clean_schedule_df(csv_path):
    # Function to read and clean a csv file containing employee schedules.
    try:
        # Load the CSV and strip whitespace from column names
        main_df = pd.read_csv(csv_path, skipinitialspace=True)

        # Clean column names (strip quotes and whitespace)
        main_df.columns = [col.strip().strip('"').strip() for col in main_df.columns]
        # print(df)

        #Handle missing values
        main_df.fillna('Unknown', inplace=True)

        # Convert date and time fields to datetime types (if applicable)
        main_df["Date"] = pd.to_datetime(main_df["Date"].astype(str), format='%Y-%m-%d', errors="coerce")
        main_df["Start Time"] = pd.to_datetime(main_df["Start Time"].astype(str), format='%H:%M', errors="coerce").dt.time
        main_df["End Time"] = pd.to_datetime(main_df["End Time"].astype(str), format='%H:%M', errors="coerce").dt.time

        # Convert Hours to float
        main_df["Hours"] = pd.to_numeric(main_df["Hours"], errors="coerce").fillna(0)
        return main_df
    
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    