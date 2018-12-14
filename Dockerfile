FROM python:3.6-stretch

# Install magenta dependencies
RUN apt-get update && \
    apt-get install -y build-essential libasound2-dev libjack-dev

# Install chromedriver dependencies
RUN apt-get update && \
    apt-get install -y chromium

# Set /usr/src/app as working dir
RUN mkdir /usr/src/app
WORKDIR /usr/src/app

# Install dev dependencies
COPY requirements-dev.txt /usr/src/app/
RUN pip install -r requirements-dev.txt

# Download checkpoint directory
RUN mkdir ckpt && \
    wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz && \
    tar --strip-components 1 -xvzf arbitrary_style_transfer.tar.gz -C ckpt/   

# Run tests
CMD python -m pytest -v && \
    python -m flake8 christmais
