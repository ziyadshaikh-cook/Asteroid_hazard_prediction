FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 5000

RUN pip install huggingface_hub
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='ZiyadShaikhcookin/asteroid-hazard-model', filename='model.pkl', local_dir='artifacts/'); \
hf_hub_download(repo_id='ZiyadShaikhcookin/asteroid-hazard-model', filename='preprocessor.pkl', local_dir='artifacts/')"

CMD ["python", "app.py"]