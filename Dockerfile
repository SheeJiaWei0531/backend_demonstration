FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tzdata \                     
    && ln -snf /usr/share/zoneinfo/Asia/Singapore /etc/localtime \
    && echo "Asia/Singapore" > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
ENV TZ=Asia/Singapore   


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 13520

CMD ["gunicorn", "-b", "0.0.0.0:13520", "app:app"]