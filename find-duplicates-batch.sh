#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --output=/scratch/project_2018026/joleskin/file_analysis/find-duplicates-%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=large

set -euo pipefail

module load python-data

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

input_path="${1:-/scratch/project_2018026/super_data}"

mkdir -p /scratch/project_2018026/joleskin/file_analysis

echo "Running duplicate finder"
echo "Node count: ${SLURM_NNODES:-1}"
echo "Tasks: ${SLURM_NTASKS:-1}"
echo "CPUs for threads: ${SLURM_CPUS_PER_TASK:-1}"
echo "Input path: $input_path"

srun --cpu-bind=cores python find-duplicates.py "$input_path" --max-threads "${SLURM_CPUS_PER_TASK:-1}" --limit 0 "${@:2}"