# Base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 libxau6 libxdmcp6 graphviz && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniforge (multi-architecture Conda)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
    -O /miniforge.sh && \
    bash /miniforge.sh -b -p /opt/miniforge && \
    rm /miniforge.sh && \
    /opt/miniforge/bin/conda clean -a

# Add Conda to PATH
ENV PATH="/opt/miniforge/bin:$PATH"

# Copy only the required files and directories
COPY environment.yml .
COPY .streamlit /app/.streamlit
COPY multi_app /app/multi_app
COPY xai_banking /app/xai_banking
COPY xaivision /app/xaivision

# Create the Conda environment
RUN conda env create -f environment.yml

# Set the shell to use Conda environment for subsequent commands
SHELL ["conda", "run", "-n", "xai_env", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Expose the port Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["conda", "run", "-n", "xai_env", "streamlit", "run", "multi_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
