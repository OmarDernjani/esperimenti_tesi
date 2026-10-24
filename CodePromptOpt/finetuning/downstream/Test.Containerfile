FROM docker.io/library/python:3.11-slim
# Public HumanEval+ check(candidate) imports NumPy.
RUN pip install --no-cache-dir numpy==1.26.4
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /work
CMD ["python", "-B", "check.py"]
