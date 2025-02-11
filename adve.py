import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# 1. Data Collection and Preprocessing
# Load your dataset (make sure to replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('Advertising Budget and Sales.csv')

# Check for missing values
if data.isnull().sum().any():
    data = data.dropna()  # Drop missing values or use data.fillna(value)

# 2. Feature Selection
X = data[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = data['Sales ($)']

# 3. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Save the trained model
joblib.dump(model, 'ad_budget_sales_model.pkl')

# 6. Visualization with Streamlit
st.title('Ad Budget vs Sales Prediction')

# Input fields for user input
tv_budget = st.number_input('TV Ad Budget ($)', min_value=0)
radio_budget = st.number_input('Radio Ad Budget ($)', min_value=0)
newspaper_budget = st.number_input('Newspaper Ad Budget ($)', min_value=0)

# Load the saved model for prediction
model = joblib.load('ad_budget_sales_model.pkl')

# Predict sales based on user input
user_input = [[tv_budget, radio_budget, newspaper_budget]]
predicted_sales = model.predict(user_input)

# Display the predicted sales
st.write(f"Predicted Sales: ${predicted_sales[0]:,.2f}")

# Plotting the Actual vs Predicted Sales
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')
ax.set_title('Actual vs Predicted Sales')

st.pyplot(fig)
