import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
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

def main():
    # Load the data
    data = pd.read_csv(sys.argv[1], sep=';')
    
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
