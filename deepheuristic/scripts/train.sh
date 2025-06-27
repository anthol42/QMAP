#!/bin/bash
#SBATCH --output=./logs/%x-%A.out
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096Mb
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=example@example.com

source venv/bin/activate

# Launch training
echo "--------------------------------------------"
echo "           Staring training...              "
echo "--------------------------------------------"

python main.py --experiment=experiment1 --config=configs/config.yml

echo "--------------------------------------------"
echo "                  Done!                     "
echo "--------------------------------------------"

