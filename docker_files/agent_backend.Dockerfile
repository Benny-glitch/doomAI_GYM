FROM python:3.9

WORKDIR /backend

COPY backend /backend

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "remote_agent:app", "--host", "0.0.0.0", "--port", "8080"]
