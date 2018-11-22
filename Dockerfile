FROM python:3.6-stretch

RUN pip install -r requirements-dev.txt
RUN python -m nltk.downloader brown

