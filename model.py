import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the 2022 data (milk_rainfall_2022.csv)
try:
    milk_rainfall_2022 = pd.read_csv('milk_rainfall_2022.csv', delimiter=',', encoding='utf-8')
    print(milk_rainfall_2022.head())
except Exception as e:
    print(f"Error loading the CSV file: {e}")

# Step 2: Clean and prepare the data
milk_rainfall_2022['Rainfall'] = pd.to_numeric(milk_rainfall_2022['Rainfall'], errors='coerce')
milk_rainfall_2022['Litres per cow'] = pd.to_numeric(milk_rainfall_2022['Litres per cow'], errors='coerce')

# Step 3: Define the features (X) and target (y) for 2022 data
X_train = milk_rainfall_2022[['Rainfall']]
y_train = milk_rainfall_2022['Litres per cow']

# Step 4: Initialize and train the XGBoost model using 2022 data
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Step 5: Create a DataFrame with the actual 2023 data
data_2023 = pd.DataFrame({
    'Year': [2023] * 12,
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Milk (Million Litres)': [180420000, 380980000, 801410000, 1028030000, 1159040000, 1045500000, 1014500000, 909100000, 763700000, 588670000, 381660000, 205320000],
    'Litres per cow': [109.57, 231.37, 486.71, 624.33, 703.90, 634.94, 616.12, 552.11, 463.80, 357.51, 231.79, 124.69],
    'Rainfall': [104.11, 31.28, 146.18, 70.97, 41.97, 63.37, 158.34, 115.21, 128.83, 158.07, 102.59, 149.28]
})

# Step 6: Use the 2023 rainfall data for prediction
X_test = data_2023[['Rainfall']]
predicted_yield_2023 = model.predict(X_test)

# Step 7: Compare predicted vs actual yield for 2023
actual_yield_2023 = data_2023['Litres per cow'].values
months = data_2023['Month'].values

# Step 8: Plot the predicted vs actual values for 2023
plt.figure(figsize=(10, 6))
plt.plot(months, actual_yield_2023, label='Actual Yield (Litres per cow)', marker='o')
plt.plot(months, predicted_yield_2023, label='Predicted Yield (Litres per cow)', marker='x')
plt.title('Predicted vs Actual Milk Yield per Cow for 2023')
plt.xlabel('Month')
plt.ylabel('Litres per cow')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 9: Evaluate the model's performance for 2023
rmse_2023 = mean_squared_error(actual_yield_2023, predicted_yield_2023, squared=False)
print(f"RMSE for 2023 Predictions: {rmse_2023}")
