# MindBigData 2023

## Environment Setup

### Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- WSL2 (Windows Subsystem for Linux)
- NVIDIA GPU

### Run Jupyter and Terminal Together (Recommended)

Start Jupyter in a detached container:

```bash
docker run --gpus all -d --rm \
  --name mbd2023 \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

The dataset will be downloaded to `data/` inside the project folder on first run and reused on subsequent runs.

Get the login URL/token from logs:

```bash
docker logs -f mbd2023
```

Open an interactive shell in the same running container:

```bash
docker exec -it mbd2023 bash
```

Stop the container when done:

```bash
docker stop mbd2023
```

### Docker Image

| Image | Version |
|-------|---------|
| `nvcr.io/nvidia/pytorch` | `26.02-py3` |

## Install Dependencies

After opening a shell in the running container, install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```