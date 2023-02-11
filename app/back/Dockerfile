FROM tiangolo/uvicorn-gunicorn:python3.9

RUN  apt-get update && apt-get install postgresql libpq-dev -y

RUN mkdir /mas_calculator

COPY requirements.txt /mas_calculator

WORKDIR /mas_calculator

RUN pip install -r requirements.txt

COPY . /mas_calculator

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]