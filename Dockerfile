FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
