#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=vesta
#SBATCH --gres=gpu:Tesla-K80:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=./slogs/slurm-%j.out
module load vesta cuda/10.2
# Script to translate wmt19 paraphrases using both hub and local NMT models
set -e

# usage function
usage() {
  cat <<EOF
Usage: translate_wmt19_paraphrases_de_en.sh [-h|--help] [glob]
Translate WMT19 paraphrases using both torch-hub and local models

Optional arguments:
  -h, --help   Show this help message and exit
  glob <glob>  Glob for finding local NMT model checkpoints, defaults to
               "./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573/checkpoint_best.pt"
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
translate_wmt19_paraphrases_de_en() {
  local glob="${1:-"./models/transformer_vaswani_wmt_en_de_big.wmt16.de-en.1594228573/checkpoint_best.pt"}"
  python3 -m src.translate_wmt19_paraphrases_de_en --checkpoints-glob "$glob"
}

# execute function
check_help "$@"
translate_wmt19_paraphrases_de_en "$@"
