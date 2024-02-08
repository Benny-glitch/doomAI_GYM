FROM python:3.9

WORKDIR /backend

COPY backend/basic_agent /backend/basic_agent

COPY doomGame /backend/doomGame

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

CMD ["python", "/backend/basic_agent/gameTrainer.py"]




