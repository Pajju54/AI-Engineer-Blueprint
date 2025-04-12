import pandas as pd
from ydata_profiling import ProfileReport
# from load import read_config, load_cleaned_data
from pathlib import Path
from logger import logger


def generate_profile_report(df: pd.DataFrame, save_path: str = "data_profile_report.html"):
    # config = read_config()
    # df = load_cleaned_data(config)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating data profile report...")
    profile = ProfileReport(df, title="CSV Samurai Data Profile Report",explorative=True)
    profile.to_file(save_path)
    logger.info(f"Profile report saved to {save_path}")