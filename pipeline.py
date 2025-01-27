import csv
import joblib
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_execution.log')
    ]
)
logger = logging.getLogger(__name__)

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.columns_to_drop = ['CustomerId', 'Surname', 'Exited', 'HasCrCard']
        self.numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance',
                              'NumOfProducts', 'EstimatedSalary']
        
    def transform(self, X):
        logger.info("Starting data transformation process")
        
        try:
            logger.debug("Creating copy of input data")
            X = X.copy()
            
            # Dropping unnecessary columns
            logger.debug(f"Dropping columns: {self.columns_to_drop}")
            X = X.drop(columns=self.columns_to_drop, errors='ignore')
            
            # Transforming gender column
            logger.debug("Transforming gender column to binary values")
            X['Male'] = X['Gender'].map({'Male': 1, 'Female': 0})
            X = X.drop(columns=['Gender'])
            
            # Creating geographic dummies
            logger.debug("Creating dummy variables for geography column")
            X = pd.get_dummies(X, columns=['Geography'], prefix='Geo', dtype=int)
            
            logger.info("Data transformation completed successfully")
            return X
            
        except Exception as e:
            logger.error(f"Error during data transformation: {str(e)}", exc_info=True)
            raise

def create_pipeline():
    logger.info("Creating the pipeline configuration")
    
    try:
        # Create the pipeline
        pipe_steps = [
            ('transformer', DataTransformer()),
            ('model', joblib.load('scalers_n_models/lightgbm.pkl'))
        ]
        
        pipeline = Pipeline(pipe_steps)
        logger.info("Pipeline created successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}", exc_info=True)
        raise

def read_csv_with_unknown_separator(file_path):
    logger.info(f"Attempting to read CSV file at path: {file_path}")
    
    try:
        # Detecting the separator
        with open(file_path, 'r') as file:
            sample = file.read(1024)
            dialect = csv.Sniffer().sniff(sample)
        
        # Loading the data with detected separator
        logger.debug(f"Detected CSV delimiter: {dialect.delimiter}")
        df = pd.read_csv(file_path, sep=dialect.delimiter)
        logger.info("CSV file loaded successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}", exc_info=True)
        raise

def main():
    logger.info("Starting script execution")
    
    try:
        # Load the data
        logger.info("Attempting to read input data file")
        data = read_csv_with_unknown_separator(sys.argv[1])
        
        # Create and load the pipeline
        logger.info("Initializing pipeline creation")
        pipeline = create_pipeline()
        
        # Making predictions
        predictions = pipeline.predict(data.iloc[:, 1:])
        logger.info(f"Successfully generated predictions for {len(predictions)} rows")
        
        # Saving predictions
        result_df = pd.DataFrame({
            'rowNumber': data['RowNumber'],
            'predictedValues': predictions
        })
        logger.info("Saving prediction results to prediction.csv")
        result_df.to_csv('prediction.csv', index=False)
        
        logger.info("Script execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
