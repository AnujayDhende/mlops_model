# ml_model.py
# This is a simple linear regression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

data = {'feature1': [1, 2, 3, 4, 5], 'target': [2, 4, 5, 4, 5]}
df = pd.DataFrame(data)

X = df[['feature1']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

model_filename = 'linear_regression_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained model saved as {model_filename}")

if __name__ == "__main__":
    print("ML model script executed.")
