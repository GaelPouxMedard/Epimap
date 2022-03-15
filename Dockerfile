###########################################
# Actual Image without build dependencies #
###########################################
FROM python:3.10
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.freeze /app/
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.freeze && pip install waitress
ENV FLASK_APP=controller.py
ENV FLASK_ENV=development
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
COPY Website /app/
RUN chown -R www-data:www-data /app
USER www-data
EXPOSE 5000
ENTRYPOINT ["/usr/local/bin/waitress-serve", "--port=5000",  "wsgi:app" ]
