# anemia_predict.py

import joblib
import pandas as pd
import json

# Load the trained model from the file
clf = joblib.load('anemia_model.joblib')

def predict_anemia(data):
    input_df = pd.DataFrame([data], columns=['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT /mm3', 'HGB'])
    prediction = clf.predict(input_df)
    return 'Anemia' if prediction == 1 else 'No Anemia'

# Load data from JSON file
with open('anemia_test.json', 'r') as f:
    data = json.load(f)

# Make prediction
print(predict_anemia(data))
