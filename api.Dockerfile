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

# Copy the environment.yml file into the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Set the shell to use Conda environment for subsequent commands
SHELL ["conda", "run", "-n", "xai_env", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy application code into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Run the FastAPI application using Conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "xai_env", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
