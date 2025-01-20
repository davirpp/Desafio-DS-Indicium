import pandas as pd
import joblib
import sys


# Load the data
data = pd.read_csv(sys.argv[1], sep=';')

# Load scaler and model
scaler = joblib.load('scalers_n_models/standard_scaler.pkl')
model = joblib.load('scalers_n_models/lightgbm.pkl')

columns_to_drop = ['CustomerId', 'Surname', 'Exited', 'HasCrCard']
numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                   'NumOfProducts', 'EstimatedSalary']

data = data.drop(columns=columns_to_drop, errors='ignore')

data['Male'] = data['Gender'].map({'Male': 1, 'Female': 0})
data.drop(columns=['Gender'], inplace=True)
data = pd.get_dummies(data, columns=['Geography'], prefix='Geo', dtype=int)

data_scaled = data.copy()
data_scaled[numeric_columns] = scaler.transform(data[numeric_columns])

predictions = model.predict(data_scaled.iloc[:, 1:])

result_df = pd.DataFrame({'rowNumber': data['RowNumber'], 'predictedValues': predictions})
result_df.to_csv('prediction.csv', index=False)

