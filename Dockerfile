FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ACE/ ./ACE/
COPY DistributedACE/ ./DistributedACE/
COPY run.py .
COPY app.py .
COPY config.template.json .

RUN cp config.template.json config.json

ENTRYPOINT ["python", "run.py"]
