clean_setup:
	pyenv virtualenv-delete -f $(shell cat .python-version)
	pyenv virtualenv $(shell cat .python-base-version) $(shell cat .python-version)
setup:
	pip install -r requirements.txt
update_dependencies:
	pip-compile --allow-unsafe --no-annotate requirements.in
	pip install -r requirements.txt
