#!/bin/bash
#SBATCH -A uoa04074
#SBATCH --job-name=mpi_py
#SBATCH --output=mpi_py_%j.log
#SBATCH -e slurm-%x.%A-%a.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=4GB

module load Python/3.11.3-gimkl-2022a
module load intel/2022a

mpirun python mpi_python.py

