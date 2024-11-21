# Explainable AI (XAI)

## Introduction

This repository contains explainable AI pipelines and functions that provide insightful visualizations and explanations for models across various domains, including computer vision and financial applications.

The tools and algorithms offered in this repository aim to enhance transparency in AI models by identifying and explaining feature contributions, uncovering patterns, and providing a deeper understanding of model behavior.

## How to Run the Applications

### Individual Apps
You can run the vision-focused or banking-focused apps independently:

1. **Banking Application**:
   ```bash
   streamlit run xai_banking/app.py
   ```
2. **Vision Application**:
   ```bash
   streamlit run main_page.py
   ```

### Multipage App
If you want to run a combined app for both vision and banking functionalities, use the multipage application:
    ```bash
    streamlit run multi_app/main.py
    ```

The multipage app provides a single interface to select between Banking Pilot and Fmake Pilot, dynamically loading the respective application.

## Environment Setup
To ensure a smooth setup and execution of the repository, follow these steps:

1. Clone the Repository
Clone this repository to your local machine:

    ```bash
    git clone <repository-url>
    cd <repository-name> 
    ```

2. Set Up Conda Environment
Create a Conda environment with all necessary dependencies:

Create the environment:
    ```bash
    conda create --name xai_env python=3.11 -y
    ```
Activate the environment:
    ```bash
    conda activate xai_env
    ```
Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## General Use Tools

The general use tools that are currently offered in the repository are listed in the table below.


| Tool                       | Description                                                                                  | Applicable Domain         |
|----------------------------|----------------------------------------------------------------------------------------------|---------------------------|
| Model Details              | Visualize the model architecture with details about total parameters and number of operations. | Vision, Banking           |
| Sample Details             | Display input with the estimated output of the model.                                        | Vision, Banking           |
| Convolution Layers Activation | Display the values of the convolutional layers stacked together.                             | Vision                    |
| Independent Components     | Split the input into independent components to identify patterns.                            | Vision                    |
| Preprocessed Data Overview | Visualize and inspect preprocessed data, such as encoded features or transformed datasets.    | Banking                   |
| Feature Contributions      | Analyze the impact of specific features (e.g., categorical or numerical) on model predictions.| Banking                   |


## Algorithms

The XAI algorithms currently offered in the repository are listed in the table below:

| Tool                       | Description                                                                                  | Applicable Domain         |
|----------------------------|----------------------------------------------------------------------------------------------|---------------------------|
| Model Details              | Visualize the model architecture with details about total parameters and number of operations. | Vision, Banking           |
| Sample Details             | Display input with the estimated output of the model.                                        | Vision, Banking           |
| Convolution Layers Activation | Display the values of the convolutional layers stacked together.                             | Vision                    |
| Independent Components     | Split the input into independent components to identify patterns.                            | Vision                    |
| Preprocessed Data Overview | Visualize and inspect preprocessed data, such as encoded features or transformed datasets.    | Banking                   |
| Feature Contributions      | Analyze the impact of specific features (e.g., categorical or numerical) on model predictions.| Banking                   |


## Banking-Specific Features

The repository now includes additional features and tools tailored for financial and banking applications:

### Tools

| Tool                     | Description                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------|
| Lime Explanation Viewer  | Visualize feature contributions for individual predictions using LIME.                      |
| SHAP Explanation Viewer  | Display global and local feature contributions using SHAP visualizations.                   |
| Top Features Viewer      | Highlight the top contributing features for a given prediction.                             |
| Data Preprocessing Tools | Preprocess and encode financial datasets for explainability tasks.                          |

### Visualizations

| Visualization          | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| Bar Chart (LIME)       | Display the contribution of each feature for an individual prediction using LIME.               |
| SHAP Summary Plot      | Visualize the global feature importance and contribution distributions.                         |
| SHAP Waterfall Plot    | Show the breakdown of individual feature contributions for a specific instance.                 |
| SHAP Heatmap Plot      | Highlight patterns and clusters of feature contributions across multiple data points.           |
| SHAP Beeswarm Plot     | Combine feature importance and distribution across data points for global insights.             |


## Applications
This repository supports two primary domains:

1. Vision Models:

- Explainability for image classification models.
- Focus on pixel-level contributions and visual feature activations.
- Tools: Vision SHAP, Integrated Gradients, DeepLift.

2. Banking Models:

- Explainability for financial and tabular data models.
- Feature-level analysis for predictions, such as fraud detection and credit scoring.
- Tools: LIME, SHAP, Preprocessed Data Overview, Feature Contributions.


## How to Use
1. Select the Application:

- For vision tasks, use tools like Vision SHAP and Integrated Gradients.
- For banking tasks, use tools like LIME and SHAP Explanation Viewers.

2. Visualize and Interpret:

- Explore detailed visualizations, such as SHAP Summary and Waterfall plots for banking models, or pixel-level explanations for vision models.

3. Extend the Framework:

Add custom explainability tools or algorithms for other domains as needed.