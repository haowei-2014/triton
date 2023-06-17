FROM nvcr.io/nvidia/tritonserver:23.03-py3

COPY model_repository /models

CMD tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=15728640 --log-verbose=1