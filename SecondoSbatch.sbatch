#!/bin/bash
#
# [prova.sbatch]
#
#SBATCH --job-name=gan_insect_train        # Job name
#SBATCH --output=log_gan.out            # Output file name 
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --time=00:30:00               # Set max runtime of job
#SBATCH --account=nct_323 --qos=acc_training

#SBATCH --nodes=1                # All GPUs on one node
#SBATCH --ntasks=1                 # 4 processes (one per GPU)
#SBATCH --cpus-per-task=20 --partition=acc --gres=gpu:1
#SBATCH --ntasks-per-node=1         # 4 tasks on 1 node (multi-GPU same node)

module purge                          # Start with a clean environment
#module load intel mkl python
source venv/bin/activate 
module load intel gcc mkl impi hdf5 python/3.12.1-gcc

start_time=$(date +%s)

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas   --capture-range=cudaProfilerApi -x true -o profile1gpu torchrun --nnodes=1 --nproc_per_node=1 GanScript.py -t -e 3
# --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_timeout=60 \

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Job completed in $elapsed_time seconds."


