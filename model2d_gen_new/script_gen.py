import os

batchsize = 1700
# slurm_header = """#!/bin/sh
# #SBATCH -D /s/ls4/users/khokhlov/model_2d_04_2025/model2d_gen_new
# #SBATCH -o %j.out
# #SBATCH -e %j.err
# #SBATCH -t 48:10:00
# #SBATCH -p hpc4-3d
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=24

# export LD_LIBRARY_PATH=/s/ls4/users/khokhlov/rect/lib/
# cd $SLURM_SUBMIT_DIR

# module load cmake/3.19.3
# module load openmpi/default
# module load gcc/default
# """
# rect_path = "/s/ls4/users/khokhlov/rect/build/rect"

# slurm_header = ""
# rect_path = "/home/alex-kuruts/my-dir/Nauch/rect_new/rect/build/rect"

slurm_header = ""
rect_path = "OMP_NUM_THREADS=12 /home/nik/agrelov/rect/build/rect"

def write_script(script_lines, batch_idx):
    with open(f"scripts/run_compute{batch_idx}.sh", "w") as f:
        f.write("\n".join(script_lines))


def script_generator():
    dataset_dir = "dataset"
    os.makedirs("scripts", exist_ok=True)

    script_lines = [slurm_header] if slurm_header else []
    batch_idx = 1
    task_count = 0

    for dataset in sorted(os.listdir(dataset_dir)):
        configs_path = os.path.join(dataset_dir, dataset, "configs")
        script_lines.append(f"\ncd {configs_path}")

        for sample in sorted(os.listdir(configs_path)):
            sample_path = os.path.join(configs_path, sample)
            script_lines.append(f"cd {sample}")

            for conf in sorted(os.listdir(sample_path)):
                if conf.endswith(".conf"):
                    script_lines.append(f"{rect_path} {conf}")
                    task_count += 1

                    if task_count >= batchsize:
                        script_lines.append("cd ..")
                        script_lines.append("cd ../../..")
                        write_script(script_lines, batch_idx)
                        batch_idx += 1
                        task_count = 0
                        script_lines = [slurm_header] if slurm_header else []
                        script_lines.append(f"\ncd {configs_path}")
                        script_lines.append(f"cd {sample}")

            script_lines.append("cd ..")

        script_lines.append("cd ../../..")

    if len(script_lines) > (1 if slurm_header else 0):
        write_script(script_lines, batch_idx)


if __name__ == "__main__":
    script_generator()
