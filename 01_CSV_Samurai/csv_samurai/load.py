import pandas as pd
import yaml
from logger import get_logger

logger = get_logger("loader")

def read_config(path="csv_samurai/config.yaml") -> dict:
    with open(path,'r') as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded.")
    return config

def load_data(config: dict) ->pd.DataFrame:
    try:
        df = pd.read_csv(config["data_path"])
        logger.info(f"Loaded data from {config['data_path']} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading the data: {e}")
        raise

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Starting the data cleaning process...")

    null_before = df.isnull().sum()
    logger.info(f"Null values before cleaning:\n{null_before}")

    if config.get("drop_duplicates", False):
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        logger.info(f"Dropped {before - after} duplicate rows.")

    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(),inplace=True)
        logger.info("Filled the missing 'Age' with median")

    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        logger.info("Filled the missing 'Embarked' with mode") 

    if "Cabin" in df.columns:
        df.drop("Cabin", axis=1, inplace=True)
        logger.info("Dropped 'Cabin' column")

    null_after = df.isnull().sum()
    logger.info(f"Nulls after cleaning:\n{null_after}")


    logger.info("Data cleaning process completed")
    return df        
