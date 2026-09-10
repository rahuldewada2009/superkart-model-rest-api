FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the Flask app listens on this port (matches app.py and the docker run command below)
EXPOSE 7860

CMD ["python", "app.py"]
