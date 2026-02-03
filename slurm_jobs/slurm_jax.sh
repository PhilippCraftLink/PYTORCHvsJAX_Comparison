#!/bin/bash -l
#SBATCH --job-name=JAX_ResNet_CiFAR         # Name des Jobs
#SBATCH --partition=main                # Partition (von Uni vorgegeben)
#SBATCH --time=01:00:00                 # maximale Laufzeit
#SBATCH --cpus-per-task=8               # Anzahl CPU-Kerne
#SBATCH --mem=32G                       # Arbeitsspeicher
#SBATCH --gres=gpu:1                    # Anzahl GPUs
#SBATCH --output=%x-%j.out              # stdout → my_training_job-12345.out
#SBATCH --error=%x-%j.err               # stderr → my_training_job-12345.err

module load CUDA/12.0
module load Anaconda3
export WANDB_API_KEY=""

# Python-Umgebung aktivieren
source ~/uv-env/bin/activate

# In dein Projekt wechseln
cd ~/thesis/jax_test/

wandb login $WANDB_API_KEY

# Training starten
python ResNet50_CIFAR_JAX2.py