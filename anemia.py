import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('anemia.csv')

X = df[['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT /mm3', 'HGB']]
y = df['Anemia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

def predict_anemia(data):
    input_df = pd.DataFrame([data], columns=['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT /mm3', 'HGB'])
    prediction = clf.predict(input_df)
    return 'Anemia' if prediction == 1 else 'No Anemia'

sample_data = [4.5, 36.0, 80.0, 26.5, 33.0, 14.5, 8.0, 150000, 12.0]  
print(predict_anemia(sample_data))
