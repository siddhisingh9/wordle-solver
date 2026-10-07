FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=7860

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 60 -r requirements-web.txt

COPY src ./src
COPY data ./data
RUN python -c "from wordle.solver import Solver; Solver.load()"

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn wordle.webapp:app --host 0.0.0.0 --port ${PORT}"]
