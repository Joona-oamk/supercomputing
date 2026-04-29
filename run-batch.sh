#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --output=/scratch/project_2018026/joleskin/file_analysis/analysis-%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=large

set -euo pipefail

module load python-data

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

input_path="${1:-/scratch/project_2018026/super_data}"
report_path="${2:-/scratch/project_2018026/joleskin/file_analysis/analysis-${SLURM_JOB_ID}.json}"

mkdir -p "$(dirname "$report_path")"

echo "Running hybrid MPI + multiprocessing analysis"
echo "Nodes: ${SLURM_NNODES:-1}"
echo "MPI ranks: ${SLURM_NTASKS:-1}"
echo "CPUs per rank: ${SLURM_CPUS_PER_TASK:-1}"
echo "Input path: $input_path"
echo "Report path: $report_path"

srun --cpu-bind=cores python analyze.py "$input_path" --progress-seconds 15 --report-json "$report_path"
