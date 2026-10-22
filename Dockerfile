FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deepresearch/ ./deepresearch/
COPY ui/ ./ui/
RUN touch chainlit.md  # empty readme: no welcome panel in the UI
COPY .chainlit/config.toml .chainlit/config.toml

EXPOSE 8000

CMD ["chainlit", "run", "ui/app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]
