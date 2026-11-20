FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY meal_planner_agent ./meal_planner_agent
COPY evals ./evals
COPY app_runner.py ./

RUN useradd --create-home --uid 10001 agentuser \
    && mkdir -p /app/data \
    && chown -R agentuser:agentuser /app
USER agentuser

EXPOSE 8080

CMD ["uvicorn", "app_runner:api", "--host", "0.0.0.0", "--port", "8080"]
