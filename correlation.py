import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os

# Set your project name
WANDB_PROJECT = "da6401_assignment2"  # Change this to your project name
OUTPUT_DIR = "./correlation_results"
METRICS = ["val_acc", "val_loss"]  # Metrics to analyze

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize wandb API
    api = wandb.Api()
    
    print(f"Fetching runs from project {WANDB_PROJECT}...")
    
    # Get all runs from the project
    runs = api.runs(WANDB_PROJECT)
    
    # Extract hyperparameters and metrics
    data = []
    for run in runs:
        if run.state != "finished":
            continue
            
        # Get config and summary
        config = run.config
        summary = run.summary._json_dict
        
        # Combine into a single row
        row = {}
        
        # Add hyperparameters
        for key, value in config.items():
            # Skip internal wandb keys
            if key.startswith('_'):
                continue
                
            # Handle different types
            if isinstance(value, (int, float, bool, str)):
                row[key] = value
            elif isinstance(value, list) and len(value) > 0:
                # For lists like filter_sizes, use the first element
                row[key] = str(value)
        
        # Add metrics
        for metric in METRICS:
            if metric in summary:
                row[metric] = summary[metric]
        
        data.append(row)
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    print(f"Processed {len(df)} completed runs")
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
    
    # Calculate correlation with metrics
    correlations = {}
    for metric in METRICS:
        if metric not in df.columns:
            print(f"Metric {metric} not found in data")
            continue
            
        # Keep only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['number'])
        if metric in numeric_df.columns:
            # Calculate correlation with the metric
            corr = numeric_df.corrwith(numeric_df[metric]).sort_values(key=abs, ascending=False)
            correlations[metric] = corr
    
    # Create correlation table
    if correlations:
        # Combine all metrics
        corr_df = pd.DataFrame(correlations)
        
        # Save correlation table
        corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_table.csv"))
        print(f"Saved correlation table to {os.path.join(OUTPUT_DIR, 'correlation_table.csv')}")
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = corr_df.head(15)  # Top 15 correlated parameters
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Hyperparameter Correlation with Performance Metrics')
        plt.tight_layout()
        
        # Save the figure
        heatmap_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        print(f"Saved correlation heatmap to {heatmap_path}")
        
        # Print top correlations
        print("\nTop correlations:")
        for metric, corr_series in correlations.items():
            print(f"\n{metric}:")
            # Print top 5 correlations
            for param, value in corr_series.head(5).items():
                if param != metric:  # Skip self-correlation
                    print(f"  {param}: {value:.3f}")
    else:
        print("No correlations could be calculated")

if __name__ == "__main__":
    main()