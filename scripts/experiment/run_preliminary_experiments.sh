#!/bin/bash
# Preliminary Experiment Runner for MindBigData2023
# Runs all 8 baseline models for 100 epochs with lr=1e-3.

echo "=========================================================="
echo " Starting 100-Epoch Baseline Benchmark"
echo "=========================================================="
echo ""

docker exec mbd2023 bash -c "python main.py --multirun \
    model=eegnet,conformer,atcnet,dgcnn,rs_stgcn,lmda_net,tsception,ctnet \
    model.lr=1e-3 \
    trainer.max_epochs=100 \
    wandb.enabled=true \
    experiment.name='baseline_100ep_lr1e-3'"

echo ""
echo "=========================================================="
echo " All experiments completed! Check your W&B Dashboard."
echo "=========================================================="
