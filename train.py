# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load Data
data = pd.read_csv('train.csv')
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# 3. Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)

# 4. Encode 'Sex'
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 5. Features and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import joblib
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")

