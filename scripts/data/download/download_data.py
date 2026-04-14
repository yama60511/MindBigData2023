"""
Download MindBigData2023_MNIST-8B dataset from HuggingFace.
Files are saved to data/raw/.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "DavidVivancos/MindBigData2023_MNIST-8B"
REPO_TYPE = "dataset"
RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"

FILES = [
    "train.csv",
    "test.csv",
    "3Dcoords.csv",
    "audiolabels/0.wav",
    "audiolabels/1.wav",
    "audiolabels/2.wav",
    "audiolabels/3.wav",
    "audiolabels/4.wav",
    "audiolabels/5.wav",
    "audiolabels/6.wav",
    "audiolabels/7.wav",
    "audiolabels/8.wav",
    "audiolabels/9.wav",
]


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        dest = RAW_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=RAW_DIR,
        )
        print(f"  -> saved to {dest}")

    print("Done.")


if __name__ == "__main__":
    download()
