"""
Part 1 of project.
Script that collects Call Option data and outputs it to ./data/
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Call_data.xlsx"

class DataHandler:
    def __init__(self):
        self.df = self._load_data(DATA_FILE)
        self.df_calls = self._load_calls(DATA_FILE)

    @staticmethod
    def _load_data(path):
        df = pd.read_excel(path, sheet_name="Data")
        df = df.set_index('Date')
        return df

    @staticmethod
    def _load_calls(path):
        df = pd.read_excel(path, sheet_name="Option_data")
        df = df.rename(columns={"px_last": "price"})
        df["maturity"] = pd.to_datetime(df["maturity"])
        return df[["maturity", "T", "strike", "price"]]

        
c = DataHandler()
print(c.df_calls)
print(c.df)