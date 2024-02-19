# Interactive Web App with Streamlit and Scikit-learn

## Installation
You need these dependencies:
```console
pip install streamlit
pip install scikit-learn
pip install matplotlib
```

## Usage - Local
Run
```console
streamlit run main.py
```

## Usage - Docker

docker run -p 8080:8080 mlimage bash

Run
```console
# Build a local docker image
docker build -t mlimage .
# Run the image
docker run -p 8080:8080 mlimage
```

