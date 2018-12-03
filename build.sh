# Install dev requirements and package
pip3 install -r requirements-dev.txt
pip3 install -e .

# Run tests
python3 -m flake8 christmais
python3 -m pytest -v
