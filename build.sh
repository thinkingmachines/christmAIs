# Script for a typical Python project
pip install -r requirements-dev.txt
python -m nltk.downloader brown
python -m flake8 christmais
python -m pytest -v
