#!/bin/bash
#
# [prova.sbatch]
#
#SBATCH --job-name=dna_feature_extract        # Job name
#SBATCH --output=log_dna.out            # Output file name 
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --time=00:30:00               # Set max runtime of job
#SBATCH --account=nct_323 --qos=acc_training
#SBATCH --cpus-per-task=20 --partition=acc --gres=gpu:1


#xxxxx --constraint=highmem # only needed for CPU



module purge                          # Start with a clean environment
module load intel mkl python
source venv/bin/activate
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_RANK=$SLURM_LOCALID
#export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas   --capture-range=cudaProfilerApi -x true -o my_profile python GanScript.py -t -e 3           # Run the script using the loaded Python module
