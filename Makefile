.PHONY: install demo download train test app
install:
	python -m pip install -e '.[app,dev]'
demo:
	banklab demo
download:
	banklab download
train:
	banklab train
test:
	python -m pytest
app:
	streamlit run app/streamlit_app.py
