import os
import numpy as np

batchsize = 100

def script_generator():
    dataset_dir = "dataset"
    configs_dir = os.path.join(dataset_dir, "configs")
    models_dir = os.path.join(dataset_dir, "models")

    if not os.path.exists(f"./scripts"):
        os.makedirs(f"./scripts", exist_ok=True)

    slurm_header = """#!/bin/sh
#SBATCH -D /s/ls4/users/khokhlov/model_2d_03_2025
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 48:10:00
#SBATCH -p hpc4-3d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

export LD_LIBRARY_PATH=/s/ls4/users/khokhlov/rect/lib/
cd $SLURM_SUBMIT_DIR

module load cmake/3.19.3
module load openmpi/default
module load gcc/default
    """
    
    script_lines = [slurm_header, f"\ncd {configs_dir}\n"]
    i = 0
    j = 0

    for config_group in sorted(os.listdir(configs_dir)):
        config_group_path = os.path.join(configs_dir, config_group)
        script_lines.append(f"cd {config_group}")
        
        config_files = sorted(f for f in os.listdir(config_group_path) if f.endswith(".conf"))
        for config_file in config_files:
            script_lines.append(f"srun /s/ls4/users/khokhlov/rect/build/rect {config_file}")
        
        script_lines.append("cd ..\n")
    
        j += 1
        if j == batchsize:
            j = 0
            i += 1
            with open(f'{"scripts/run_compute"}{i}{".sh"}', "w") as script_file:
                script_file.write("\n".join(script_lines))
            script_lines = [slurm_header, f"\ncd {configs_dir}\n"]


if __name__ == "__main__":
    script_generator()
        
