import typer
from rich.console import Console
from rich.table import Table
from load import read_config, load_data, clean_data
import os

app = typer.Typer()
console = Console()

@app.command()
def load_clean(preview: bool = True, save: bool = False, output_path: str = "cleaned_train.csv"):
    """
    Load and clean the dataset as per config.yaml.
    
    options:
    --preview / --no-preview : Show preview of the cleaned data
    --save / --no-save : Save the cleaned data to a CSV file
    --output-path : Output path for the cleaned data
    """

    config = read_config()
    df = load_data(config)
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
def profile():
    """
    Generate automated data profiling report (Coming Soon)
    """
    console.print("[bold blue] Data profiling coming soon. Stay tuned!")


if __name__ == "__main__":
    app()