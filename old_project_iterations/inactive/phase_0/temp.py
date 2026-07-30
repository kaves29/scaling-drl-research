import os
import glob
import wandb

# 1. Point to your local logs
runs_dir = "runs"

# Find all run directories in the folder
all_folders = glob.glob(os.path.join(runs_dir, "*__phase_0_validation__*"))

print(f"Found {len(all_folders)} total local folders.")

for folder_path in all_folders:
    folder_name = os.path.basename(folder_path)
    
    # 🚨 CRITICAL FILTER: Skip the SET Hopper runs that are already on WandB
    if "SET__Hopper" in folder_name and "0.01" or "0.05" in folder_name:
        print(f"⏭️ Skipping (Already on WandB): {folder_name}")
        continue
        
    # Extract structural variables out of the remaining directory names
    parts = folder_name.split("__")
    algo_type = "rigl" if "rigl" in parts[0].lower() else "set"
    env_type = "Hopper" if "hopper" in parts[1].lower() else "Reacher"
    
    print(f"\n🚀 Syncing local-only run: {folder_name}")
    
    # Initialize the run WITHOUT sync_tensorboard=True to avoid the patch crash
    run = wandb.init(
        project="sparse-ppo-drl-research",
        name=folder_name,
        group=f"phase_0/{algo_type}",
        job_type=env_type
    )
    
    # Safely find and upload all the local tfevents binary data directly to this cloud run
    tfevent_files = glob.glob(os.path.join(folder_path, "*tfevents*"))
    for f in tfevent_files:
        # Pushes the historical log file directly to your run dashboard files
        wandb.save(f, base_path=runs_dir, policy="now")
    
    # Close the run cleanly before moving to the next loop iteration
    run.finish()

print("\n🎉 Synchronization complete! All local-only baselines have been safely uploaded.")