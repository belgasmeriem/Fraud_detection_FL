FROM python:3.10-slim

WORKDIR /app

# libgomp1 requis par PyTorch pour le parallélisme CPU
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# flwr==1.5.0 : supporte start_numpy_client + root_certificates
RUN pip install --no-cache-dir \
    flwr==1.5.0 \
    torch==2.2.0 \
    pandas \
    pyarrow \
    scikit-learn \
    numpy

COPY client.py privacy.py ./
COPY models/ ./models/

CMD ["python", "client.py"]
