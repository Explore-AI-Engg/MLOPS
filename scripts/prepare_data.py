"""Create the cleaned binary-target CSV from the downloaded raw data."""

from app.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from app.data import load_raw_data, save_processed_data

if __name__ == "__main__":
    save_processed_data(load_raw_data(RAW_DATA_PATH), PROCESSED_DATA_PATH)
    print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")
