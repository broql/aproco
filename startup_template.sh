#!/bin/bash

# Install Streamlit if not already installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit not found, installing..."
    pip install streamlit==1.24.1
fi
pip install langchain==0.0.251
pip install python-dotenv==1.0.0
pip install openai==0.27.8
pip install numpy==1.25.2

export HNSWLIB_NO_NATIVE=1

apt install python3-dev
apt-get install build-essential -y

pip install chromadb==0.3.23

#### !!! FILL THESE IN WITH YOUR OWN VALUES !!! ####
export OPENAI_API_TYPE=
export OPENAI_API_VERSION=
export OPENAI_API_BASE=
export OPENAI_API_KEY=

pip install tiktoken==0.4.0
pip install pypdf==3.12.1
pip install pandas==2.0.3

# Run the Streamlit app
streamlit run gui/app.py  --server.port 8000 --server.address 0.0.0.0