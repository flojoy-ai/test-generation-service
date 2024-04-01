FROM python:3.11-slim-buster
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY ./pyproject.toml ./poetry.lock* /app/

ADD . /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
