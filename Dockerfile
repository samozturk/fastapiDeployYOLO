FROM python:3.7

WORKDIR /usr/local/src/app

COPY . .

RUN pip install --no-cache-dir -r  requirements.txt

EXPOSE 80


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
