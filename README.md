# MindBigData2023 — Brain-Computer Interface Challenge

A modular deep learning pipeline for classifying 10 digits (0–9) from 128-channel EEG signals using CNNs, Transformers, and Graph Neural Networks.

Built with **PyTorch Lightning** for training loops, **Hydra** for configuration management, and **W&B** for experiment tracking.

---

## Supported Architectures

| Model | Type |
|-------|------|
| **EEGNet** | CNN — lightweight baseline |
| **CTNet** | CNN + Transformer hybrid |
| **Conformer** | Transformer adapted for EEG |
| **ATCNet** | TCN + sliding-window attention |
| **LMDA-Net** | CNN + multi-dimensional attention |
| **TSception** | Multi-scale CNN |
| **DGCNN** | Graph Neural Network (dynamic topology) |
| **RS-STGCN** | Graph Neural Network (regional-synergy) |

---

## Environment Setup

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with CUDA support

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MindBigData2023.git
cd MindBigData2023
```

### 2. Configure W&B API key
Create a `.env` file in the project root (this file is gitignored):
```bash
echo "WANDB_API_KEY=your_api_key_here" > .env
```
Get your API key from https://wandb.ai/authorize.

### 3. Build and launch the Docker container
```bash
docker compose up -d
```
This will:
- Build the image (`mbd2023-dev`) with all dependencies pre-installed
- Mount the project at `/workspace`
- Load your W&B API key from `.env`
- Start JupyterLab at http://localhost:8888

The container is named `mbd2023`.

**Access JupyterLab:**
```bash
docker logs mbd2023 2>&1 | grep "http://127.0.0.1"
```

**Access the container terminal:**
```bash
docker exec -it mbd2023 bash
```

**Stop the container:**
```bash
docker compose down
```

### 4. Download the dataset
```bash
docker exec mbd2023 python scripts/data/download/download_data.py
```

### 5. Build the preprocessed HDF5 dataset
Run these scripts in order:
```bash
# Step 1 — chunk raw CSV by recording date
docker exec mbd2023 python scripts/data/preprocess/chunk_by_date.py

# Step 2 — scale EEG to µV, extract digit-only rows, split train/val/test
docker exec mbd2023 python scripts/data/preprocess/build_digits_scaled.py

# Step 3 — apply 1–40 Hz bandpass filter (final training-ready data)
docker exec mbd2023 python scripts/data/preprocess/build_digits_filtered.py
```

Output: `data/processed/digits_1_40hz/{train,val,test}.h5` (88,954 / 31,046 / 20,000 samples)

---

## Training

All commands run inside Docker. The entry point is `main.py`, driven entirely by Hydra.

### Default run (EEGNet)
```bash
docker exec mbd2023 bash -c "cd /workspace && python main.py"
```

### Switch model
```bash
docker exec mbd2023 bash -c "cd /workspace && python main.py model=conformer"
```

### Override hyperparameters
```bash
docker exec mbd2023 bash -c "cd /workspace && python main.py model=atcnet model.lr=5e-4 trainer.max_epochs=100"
```

### Multi-run sweep (all 8 models)
```bash
docker exec mbd2023 bash -c "cd /workspace && python main.py --multirun \
  model=eegnet,conformer,atcnet,dgcnn,rs_stgcn,lmda_net,tsception,ctnet"
```

### Quick sanity check (no GPU needed)
```bash
docker exec mbd2023 bash -c "cd /workspace && python main.py hydra=debug wandb=disabled trainer.fast_dev_run=true"
```

### Disable W&B
```bash
docker exec mbd2023 bash -c "cd /workspace && python main.py wandb=disabled"
```

---

## Configuration

Configs are organized into Hydra groups — swap any group from the CLI:

| Group | Options | Default |
|-------|---------|---------|
| `model` | `eegnet`, `conformer`, `atcnet`, `dgcnn`, `rs_stgcn`, `lmda_net`, `tsception`, `ctnet` | `eegnet` |
| `preprocessing` | `zscore`, `de_features`, `none` | `zscore` |
| `trainer` | `default` | `default` |
| `wandb` | `default`, `disabled` | `default` |
| `hydra` | `default`, `sweep`, `debug` | `default` |

**Note:** Graph models (`dgcnn`, `rs_stgcn`) automatically use DE features regardless of the `preprocessing` setting.

### Output structure
```
outputs/
  <model>/<timestamp>/          # single run
    ├── train.log
    ├── checkpoints/
    └── .hydra/
  multirun/<timestamp>/         # --multirun
    └── <model>-<job_num>/
```

### W&B run naming
- **Single run**: run name = model name (e.g. `eegnet`)
- **Multirun**: run name = `{model}-{job_num}` (e.g. `eegnet-0`), grouped by sweep timestamp

---

## Notes

- **W&B re-auth**: If W&B loses authentication, run `docker exec -it mbd2023 wandb login`.
- **GPU memory**: Reduce `data.batch_size` if you get CUDA out-of-memory errors (e.g. `data.batch_size=32`).
- **Data workers**: `num_workers=0` is safest in Docker. Increase only if you have sufficient shared memory.
