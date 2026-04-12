#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=hello
#SBATCH --output=/scratch/project_2018026/joleskin-analysis.log
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=test

module load python-data

input_path="${1:-super_data}"
python analyze.py "$input_path"