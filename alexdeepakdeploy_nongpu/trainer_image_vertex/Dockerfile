FROM gcr.io/deeplearning-platform-release/tf-gpu.2-8
RUN pip install -U fire cloudml-hypertune
WORKDIR /app
COPY model.py .

ENTRYPOINT ["python", "model.py"]