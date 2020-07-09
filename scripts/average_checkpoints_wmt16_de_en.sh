#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=generic
#SBATCH --mem=48G
#SBATCH --ntasks=5
#SBATCH --output=./slogs/slurm-%j.out
module load generic
# Average last N checkpoints from training directory
set -e

# usage function
usage() {
  cat <<EOF
Usage: average_checkpoints_wmt16_de_en [-h|--help] model_directory [number]
Average the last N checkpoints from a model directory

Optional arguments:
  -h, --help              Show this help message and exit
  number <int>            Number of last checkpoints to average,
                          defaults to 10.

Required arguments:
  model_directory <path>  Path to directory containing model checkpoints
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 1
    fi
  done
}

# define function
average_checkpoints_wmt16_de_en() {
  # declare variables
  local direct="$1" number="${2:-10}"
  [ ! -d "$direct" ] && usage && exit 1
  # average checkpoints
  python3 -m src.average_checkpoints_fairseq \
    --input-directory "$direct" \
    --num-epoch-checkpoints "$number"
}

# execute function
check_help "$@"
average_checkpoints_wmt16_de_en "$@"
