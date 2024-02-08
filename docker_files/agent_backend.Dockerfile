FROM python:3.9

WORKDIR /backend

COPY backend /backend

COPY doomGame /doomGame

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt



