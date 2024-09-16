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

# Step 3: Function to train and predict for each month using only that month's historical data
def train_and_predict_for_month(month, year_2023_rainfall):
    # Filter data for the specific month across all years (excluding 2023)
    month_data = milk_rainfall_2022[milk_rainfall_2022['Month'] == month]
    X_train = month_data[['Rainfall']]
    y_train = month_data['Litres per cow']
    
    # Initialize and train the XGBoost model for the specific month
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # Predict the 2023 milk yield per cow for the given month using 2023 rainfall
    predicted_yield = model.predict([[year_2023_rainfall]])[0]
    
    return predicted_yield

# Step 4: Create a DataFrame with the actual 2023 data
data_2023 = pd.DataFrame({
    'Year': [2023] * 12,
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Milk (Million Litres)': [180420000, 380980000, 801410000, 1028030000, 1159040000, 1045500000, 1014500000, 909100000, 763700000, 588670000, 381660000, 205320000],
    'Litres per cow': [109.57, 231.37, 486.71, 624.33, 703.90, 634.94, 616.12, 552.11, 463.80, 357.51, 231.79, 124.69],
    'Rainfall': [104.11, 31.28, 146.18, 70.97, 41.97, 63.37, 158.34, 115.21, 128.83, 158.07, 102.59, 149.28]
})

# Step 5: Use the month-specific models to predict milk yield for 2023
predicted_yield_2023 = []
actual_yield_2023 = []
months_2023 = data_2023['Month'].values
rainfall_2023 = data_2023['Rainfall'].values

for i, month in enumerate(months_2023):
    actual_yield = data_2023.loc[data_2023['Month'] == month, 'Litres per cow'].values[0]
    actual_yield_2023.append(actual_yield)
    
    # Predict yield for the month using only that month's historical data
    predicted_yield = train_and_predict_for_month(month, rainfall_2023[i])
    predicted_yield_2023.append(predicted_yield)

# Step 6: Plot the predicted vs actual values for 2023
plt.figure(figsize=(10, 6))
plt.plot(months_2023, actual_yield_2023, label='Actual Yield (Litres per cow)', marker='o')
plt.plot(months_2023, predicted_yield_2023, label='Predicted Yield (Litres per cow)', marker='x')
plt.title('Predicted vs Actual Milk Yield per Cow for 2023 (By Month)')
plt.xlabel('Month')
plt.ylabel('Litres per cow')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 7: Evaluate the model's performance for 2023
rmse_2023 = mean_squared_error(actual_yield_2023, predicted_yield_2023, squared=False)
print(f"RMSE for 2023 Predictions: {rmse_2023}")
