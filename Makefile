.PHONY: lint

lint:
    flake8 src
    black src --check
