import os

def link_slurm_logs(slurm_log_file: str, out_file: str) -> None:
    if os.path.isfile(out_file):
        os.remove(out_file)
    if os.path.isdir(os.path.dirname(slurm_log_file)):
        if os.path.isfile(slurm_log_file):
            os.symlink(slurm_log_file, out_file)
            return None
    print(f"Unable to link SLURM logs; in file [{slurm_log_file}] out file [{out_file}]")
