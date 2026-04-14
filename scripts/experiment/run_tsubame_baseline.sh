#!/bin/bash
# TSUBAME Baseline Experiment
# Runs all 8 models with scaled-up architectures on 4x H100 GPUs.
#
# Usage:
#   bash scripts/experiment/run_tsubame_baseline.sh
#
# Config summary:
#   - Trainer:  4 GPUs, DDP, AdamW, CosineAnnealing, 10-epoch warmup
#   - Data:     batch_size=256 per GPU (effective 1024), num_workers=8
#   - Models:   *_tsubame variants with increased hidden dimensions
#   - Training: 200 epochs max, patience=30

echo "=========================================================="
echo " TSUBAME — Scaled Baseline Benchmark"
echo " 4 GPUs | DDP | 200 epochs | AdamW + CosineAnnealing"
echo "=========================================================="
echo ""

python main.py --multirun \
    trainer=tsubame \
    data=mindbigdata_tsubame \
    model=eegnet_tsubame,conformer_tsubame,atcnet_tsubame,dgcnn_tsubame,rs_stgcn_tsubame,lmda_net_tsubame,tsception_tsubame,ctnet_tsubame \
    wandb.enabled=true \
    experiment.name='tsubame_baseline'

echo ""
echo "=========================================================="
echo " All experiments completed! Check your W&B Dashboard."
echo "=========================================================="
