# Install dev requirements and package
pip3 install -r requirements-dev.txt
pip3 install -e .

# Get brown corpus from nltk
python3 -m nltk.downloader brown

# Run tests
python3 -m flake8 christmais
python3 -m pytest -v
