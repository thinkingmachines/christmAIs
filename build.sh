# Install dev requirements and package
pip3 install -r requirements-dev.txt
pip3 install -e .

# Install magenta (without conda)
sudo apt-get install -y build-essential libasound2-dev libjack-dev
pip install magenta

# Run tests
python3 -m flake8 christmais
python3 -m pytest -v
