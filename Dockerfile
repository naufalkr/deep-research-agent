FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deepresearch/ ./deepresearch/
COPY ui/ ./ui/
COPY chainlit.md .

EXPOSE 8000

CMD ["chainlit", "run", "ui/app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]
