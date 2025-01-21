import csv
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sys

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['CustomerId', 'Surname', 'Exited', 'HasCrCard']
        self.numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                              'NumOfProducts', 'EstimatedSalary']
    
    def transform(self, X):
        # Create a copy of the input data
        X = X.copy()
        
        # Drop unnecessary columns
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Transform gender
        X['Male'] = X['Gender'].map({'Male': 1, 'Female': 0})
        X = X.drop(columns=['Gender'])
        
        # Create geography dummies
        X = pd.get_dummies(X, columns=['Geography'], prefix='Geo', dtype=int)
        
        return X

def create_pipeline():
    # Create the pipeline
    pipeline = Pipeline([
        ('transformer', DataTransformer()),
        ('model', joblib.load('scalers_n_models\lightgbm.pkl'))
    ])
    
    return pipeline

def read_csv_with_unknown_separator(file_path):
    # Automatically detect the separator
    with open(file_path, 'r') as file:
        sample = file.read(1024)  # Read a sample of the file
        dialect = csv.Sniffer().sniff(sample)
    
    # Load the file using the detected separator
    df = pd.read_csv(file_path, sep=dialect.delimiter)
    return df


def main():
    # Load the data
    data = read_csv_with_unknown_separator(sys.argv[1])
    
    # Create and load the pipeline
    pipeline = create_pipeline()
    
    # Make predictions
    predictions = pipeline.predict(data.iloc[:, 1:])
    
    # Save predictions
    result_df = pd.DataFrame({
        'rowNumber': data['RowNumber'],
        'predictedValues': predictions
    })
    result_df.to_csv('prediction.csv', index=False)

if __name__ == "__main__":
    main()
