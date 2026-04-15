#!/bin/bash
# Baseline Experiment Runner for MindBigData2023
# Runs all 8 models for 50 epochs, no early stopping.

echo "=========================================================="
echo " Starting 50-Epoch Baseline Benchmark"
echo "=========================================================="
echo ""

cd /workspace && python main.py --multirun \
    model=eegnet,conformer,atcnet,dgcnn,rs_stgcn,lmda_net,tsception,ctnet \
    trainer.max_epochs=50 \
    trainer.patience=0 \
    data.batch_size=64 \
    wandb.enabled=true \
    experiment.name='baseline'

echo ""
echo "=========================================================="
echo " All experiments completed! Check your W&B Dashboard."
echo "=========================================================="
