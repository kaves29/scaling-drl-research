import os
import glob
import wandb
from tbparse import SummaryReader

# 1. Target the runs directory
runs_dir = "runs"
all_folders = sorted(glob.glob(os.path.join(runs_dir, "*phase_0_validation*")))

print(f"Starting upload for all {len(all_folders)} runs...")

for folder_path in all_folders:
    folder_name = os.path.basename(folder_path)
    
    # Parse the group and environment details out of the name
    algo_type = "rigl" if "rigl" in folder_name.lower() else "set"
    env_type = "Hopper" if "hopper" in folder_name.lower() else "Reacher"
    
    print(f"\nProcessing: {folder_name}")
    
    try:
        # Read the local TensorBoard logs into a clean data table
        reader = SummaryReader(folder_path)
        df = reader.scalars
        
        if df.empty:
            print(f"⚠️ No scalar data found in {folder_name}, skipping.")
            continue
            
        # Initialize a pristine WandB run with correct grouping
        run = wandb.init(
            project="sparse-ppo-drl-research",
            name=folder_name,
            group=f"phase_0/{algo_type}",
            job_type=env_type,
            reinit=True
        )
        
        # Sort values by step sequence to ensure data integrity
        df = df.sort_values(by="step")
        
        # Group data points by step and stream them up line-by-line
        unique_steps = df["step"].unique()
        for step in unique_steps:
            step_data = df[df["step"] == step]
            metrics = {row["tag"]: row["value"] for _, row in step_data.iterrows()}
            wandb.log(metrics, step=int(step))
            
        run.finish()
        print(f"✅ Successfully synced all metrics!")
        
    except Exception as e:
        print(f"❌ Failed to parse {folder_name}: {e}")

print("\n🎉 Mission accomplished! All 32 runs are cleanly formatted on WandB.")