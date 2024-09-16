import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the milk and rainfall data
try:
    milk_rainfall_data = pd.read_csv('milk_rainfall.csv', delimiter=',', encoding='utf-8')
    print(milk_rainfall_data.head())
except Exception as e:
    print(f"Error loading the CSV file: {e}")

# Step 2: Clean and prepare the data
milk_rainfall_data['Litres per cow'] = pd.to_numeric(milk_rainfall_data['Litres per cow'], errors='coerce')

# Step 3: Convert the 'Month' column to a categorical variable to ensure correct ordering
milk_rainfall_data['Month'] = pd.Categorical(milk_rainfall_data['Month'], 
                                             categories=['January', 'February', 'March', 'April', 'May', 'June', 
                                                         'July', 'August', 'September', 'October', 'November', 'December'], 
                                             ordered=True)

# Step 4: Set years to be highlighted and customized
highlight_years = [2020, 2021, 2022, 2023]

# Step 5: Create a plot for 'Litres per cow' across months for each year
plt.figure(figsize=(12, 8))

# Group the data by Year and plot each year's data
for year, year_data in milk_rainfall_data.groupby('Year'):
    if year in highlight_years:
        # Plot with color and thicker lines for highlighted years
        plt.plot(year_data['Month'], year_data['Litres per cow'], label=str(year), linewidth=2.5)
    else:
        # Plot all other years in grey
        plt.plot(year_data['Month'], year_data['Litres per cow'], color='grey', linewidth=1, alpha=0.6)

# Step 6: Customize the plot
plt.title('Milk Yield per Cow by Month for Each Year')
plt.xlabel('Month')
plt.ylabel('Litres per cow')
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
plt.grid(True)
plt.xticks(rotation=45)  # Rotate month names for better readability
plt.tight_layout()

# Step 7: Show the plot
plt.show()
