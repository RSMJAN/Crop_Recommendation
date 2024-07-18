import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("crop_recommendation_data.csv")

X = data[['Soil', 'pH', 'Rainfall', 'Location']]
y = data['Crop']

X = pd.get_dummies(X, columns=['Location', 'Soil'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

new_data_point = pd.DataFrame({
    'Soil': ['Loam'],
    'pH': [6.5],
    'Rainfall': [1000],
    'Location': ['YourLocation']
})

new_data_point = pd.get_dummies(new_data_point, columns=['Location', 'Soil'])

recommended_crop = rf_classifier.predict(new_data_point)
print(f"Recommended Crop: {recommended_crop[0]}")
