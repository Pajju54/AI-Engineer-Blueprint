import typer
from rich.console import Console
from rich.table import Table
from load import read_config, load_data , clean_data
from eda_report import generate_profile_report
from features.pipeline import build_preprocessing_pipeline
import pandas as pd
import os

raw = 'raw'
cleaned = 'cleaned'

app = typer.Typer()
console = Console()

@app.command()
def load_clean(preview: bool = True, save: bool = False, output_path: str = "data/cleaned/cleaned_train.csv"):
    """
    Load and clean the dataset as per config.yaml.
    
    options:
    --preview / --no-preview : Show preview of the cleaned data
    --save / --no-save : Save the cleaned data to a CSV file
    --output-path : Output path for the cleaned data
    """

    config = read_config()
    df = load_data(config, raw)
    cleaned_df = clean_data(df,config)

    if preview:
        console.rule("[bold green]Cleaned Data Preview")
        table = Table(show_lines=True)
        for col in cleaned_df.columns:
            table.add_column(col, style="cyan")

        for _, row in cleaned_df.head(5).iterrows():
            table.add_row(*[str(x) for x in row.tolist()])
        console.print(table)

    if save:
        cleaned_df.to_csv(output_path, index=False)
        console.print(f"[bold yellow] Cleaned data saved to {output_path}")

    console.print("[bold green] All done! Your data is cleaner than a samurai's sword!")

@app.command()
def train():
    """
    Train your mL models
    """
    console.print("[bold blue] Model training coming in Day 3. Saty tuned!")

@app.command()
def profile(
    source_path:str = typer.Option("cleaned", help="Which dataset to profile: raw or cleaned"),
    output_path: str = typer.Option("reports/eda/profile_report.html", help="Path to store the profile report")
):
    """
    Generate a automated EDA report (HTML) from raw or cleaned dataset
    """
    config = read_config()
    df = load_data(config, source_path)

    generate_profile_report(df, output_path)
    console.print(f"[bold green] Profile report saved to {output_path}")


@app.command()
def preprocess(
    source_path:str = typer.Option("cleaned", help="Which dataset to preprocess: raw or cleaned"),
    save: bool = typer.Option(False, help="Weather to save the preprocessed output as CSV"),
    output_path: str = typer.Option("data/preprocessed/preprocessed_train.csv", help="Path to store the Preprocessed CSV file")
):
    """
    Preprocess the dataset using the pipeline defined in features.py.
    """
    config = read_config()

    # Assuming build_preprocessing_pipeline() is a function that returns a preprocessing pipeline
    console.print(f"[bold yellow] Loading the data from: [/bold yellow]{source_path}")
    df = load_data(config, source_path)

    #drop irrelevent cols
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=drop_cols, errors='ignore')

    #split features and target
    X = df.drop(columns=['Survived'], errors='ignore')
    y = df['Survived']

    #build and apply preprocessing
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)

    console.print("[bold green] Preprocessing complete! [/bold green]")
    console.print(f" Transformed  shape: {X_transformed.shape}")

    if save:
        pd.DataFrame(X_transformed).to_csv(output_path, index=False)
        console.print(f"[bold blue] Saved preprocessed data to: [/bold blue] {output_path}")


if __name__ == "__main__":
    app()