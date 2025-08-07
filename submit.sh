#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_sky_big
#SBATCH --job-name=DLQPM
#SBATCH --output=Job.log.%j
#SBATCH --error=Job.err.%j

python3 DLQPM.py
