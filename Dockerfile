FROM python:3.9-slim

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . /app

CMD ["poetry", "run", "python", "script.py"]
