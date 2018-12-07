.PHONY: build
build: venv requirements.txt
	@echo "Downloading categories.txt..."
	wget https://storage.googleapis.com/tm-christmais/categories.txt
	mkdir categories && \
	    mv categories.txt categories/
	@echo "Categories stored in ./categories/categories.txt"
	@echo "Downloading checkpoint model..."
	mkdir ckpt && \
	  wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz && \
	  tar --strip-components 1 -xvzf arbitrary_style_transfer.tar.gz -C ckpt/   
	@echo "Checkpoint model stored in ./ckpt/model.ckpt"
	@echo "Downloading chromedriver..."
	mkdir webdriver && \
	    wget https://chromedriver.storage.googleapis.com/2.44/chromedriver_linux64.zip && \
	    unzip chromedriver_*.zip -d webdriver
	@echo "WebDriver stored in ./webdriver/chromedriver"
	@echo "Installing requirements..."
	venv/bin/pip-sync
	@echo "Installing christmAIs..."
	venv/bin/python3 setup.py install --user
dev: venv requirements-dev.txt
	@echo "Downloading categories.txt..."
	wget https://storage.googleapis.com/tm-christmais/categories.txt
	mkdir categories && \
	    mv categories.txt categories/
	@echo "Categories stored in ./categories/categories.txt"
	@echo "Downloading checkpoint model..."
	mkdir ckpt && \
	  wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz && \
	  tar --strip-components 1 -xvzf arbitrary_style_transfer.tar.gz -C ckpt/   
	@echo "Checkpoint model stored in ./ckpt/model.ckpt"
	@echo "Downloading chromedriver..."
	mkdir webdriver && \
	    wget https://chromedriver.storage.googleapis.com/2.44/chromedriver_linux64.zip && \
	    unzip chromedriver_*.zip -d webdriver
	@echo "WebDriver stored in ./webdriver/chromedriver"
	@echo "Installing dev requirements..."
	venv/bin/pip-sync requirements-dev.txt
venv:
	python3 -m venv venv
	venv/bin/pip3 install pip-tools
requirements.txt: requirements.in
	venv/bin/pip-compile -o requirements.txt --no-header --no-annotate requirements.in 
requirements-dev.txt: requirements-dev.in
	venv/bin/pip-compile -o requirements-dev.txt --no-header --no-annotate requirements-dev.in 
