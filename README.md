# Customer Churn Prediction Project

## Overview
This project focuses on analyzing and predicting customer churn for a bank using machine learning techniques. Churn prediction is crucial for businesses to identify customers who are likely to leave their services, enabling proactive retention strategies.

## Project Structure
- `eda.ipynb`: Exploratory Data Analysis notebook detailing the dataset insights
- `predict_churn.ipynb`: Model development and evaluation notebook
- `pipeline.py`: Production-ready pipeline for making predictions
- `requirements.txt`: Required Python packages
- `scalers_n_models/`: Directory containing trained models and scalers
  - `lightgbm.pkl`: LightGBM trained model
  - `standard_scaler.pkl`: StandardScaler fitted object
  
## How to run 
1. Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate
```

2. Install the required packages:
```sh
pip install -r requirements.txt
```
And you're good to go! You can now run the notebooks

## Making Predictions
To make predictions using the production-ready pipeline, you can use the following command:
```sh
python pipeline.py your_data.csv
```
Replace `your_data.csv` with the path to your data file. 

> Note: The data file should have the same structure as the original dataset (`Abandono_teste.csv`) used in this project.


