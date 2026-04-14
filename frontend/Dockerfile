FROM python:3.13.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY frontend /app/frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
