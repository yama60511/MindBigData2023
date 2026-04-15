#!/bin/bash
# Preliminary Experiment Runner for MindBigData2023
# Runs all 8 baseline models for 100 epochs.

echo "=========================================================="
echo " Starting 100-Epoch Baseline Benchmark"
echo "=========================================================="
echo ""

docker exec mbd2023 bash -c "python main.py --multirun \
    model=eegnet,conformer,atcnet,dgcnn,rs_stgcn,lmda_net,tsception,ctnet \
    trainer.max_epochs=100 \
    trainer.patience=20 \
    wandb.enabled=true \
    experiment.name='baseline'"

echo ""
echo "=========================================================="
echo " All experiments completed! Check your W&B Dashboard."
echo "=========================================================="
