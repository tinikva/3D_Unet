#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=unet
#SBATCH --gres=gpu:1
#SBATCH --output=3D_Unet_run_%j.log


module purge

singularity exec --nv \
	    --overlay /scratch/ns4964/practenv/nspytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python 3D_Unet.py '/scratch/ns4964/lionnet/output' 50 1e-4"