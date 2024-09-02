FROM python:3.11

WORKDIR /app
RUN apt update \
    && apt install -y \
    ffmpeg \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && rm -rf /root/.cache

COPY . /app
RUN python -m pytest tests/

CMD ["pygbag", "/app"]
