#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=vesta
#SBATCH --gres=gpu:Tesla-K80:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=./slogs/slurm-%j.out
module load vesta cuda/10.2
# Script to evaluate WMT19 translations and paraphrases
# using pre-trained paraphrase detection model(s)
set -e

# usage function
usage() {
  cat <<EOF
Usage: evaluate_paraphrase_detection_wmt19_paraphrases_de_en.sh [-h|--help] [glob]
Conduct evaluation of WMT19 paraphrases using pre-trained paraphrase
detection models

Optional arguments:
  -h, --help   Show this help message and exit
  glob <glob>  Glob for finding input json translations, defaults to
               "./predictions/*/*.json"
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
evaluate_paraphrase_detection_wmt19_paraphrases_de_en() {
  local glob="${1:-"./predictions/*/*.json"}"
  local GPU=0
  CUDA_VISIBLE_DEVICES=$GPU python3 -m src.evaluate_paraphrase_detection_wmt19_paraphrases_de_en --json-glob "$glob"
}

# execute function
check_help "$@"
evaluate_paraphrase_detection_wmt19_paraphrases_de_en "$@"
