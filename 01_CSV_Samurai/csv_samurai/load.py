import pandas as pd
import yaml
from logger import logger


def read_config(path="csv_samurai/config.yaml") -> dict:
    with open(path,'r') as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded.")
    return config

def load_data(config: dict,source_path: str) -> pd.DataFrame:
    try:
        if source_path == "raw":
            df = pd.read_csv(config["data"]["raw_path"])
            logger.info(f"Loaded raw data from {config['data']['raw_path']} with shape {df.shape}")
            return df
        elif source_path == "cleaned":
            df = pd.read_csv(config["data"]["cleaned_path"])
            logger.info(f"Loaded the cleaned data from {config['data']['cleaned_path']} with shape {df.shape}")
            return df
        else:
            logger.info("Invalid source. Choose 'raw' or 'cleaned'.")

    except Exception as e:
        logger.error(f"Error loading the data: {e}")
        raise

# def load_raw_data(config: dict) ->pd.DataFrame:
#     try:
#         df = pd.read_csv(config["data"]["raw_path"])
#         logger.info(f"Loaded raw data from {config['data']['raw_path']} with shape {df.shape}")
#         return df
#     except Exception as e:
#         logger.error(f"Error loading the data: {e}")
#         raise

# def load_cleaned_data(config: dict) -> pd.DataFrame:
#     try:
#         df = pd.read_csv(config["data"]["cleaned_path"])
#         logger.info(f"Loaded the cleaned data from {config['data']['cleaned_path']} with shape {df.shape}")
#         return df
#     except Exception as e:
#         logger.error(f"Error loading the cleaned data: {e}")
#         raise

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
