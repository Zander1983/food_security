import pandas as pd

# Load the CSV file
df = pd.read_csv('rainfall_12_stations.csv')

# Extract 'Year' and 'Month' from the 'Month' column
df['Year'] = df['Month'].str[:4]
df['Month'] = df['Month'].str[5:]

# Create a dictionary to store the sum and count of rainfall values for each Year-Month
rainfall_data = {}

# Loop through each row and accumulate rainfall values
for index, row in df.iterrows():
    year = row['Year']
    month = row['Month']
    value = row['VALUE']
    
    # Skip rows with missing rainfall values
    if pd.notna(value):
        key = (year, month)
        if key not in rainfall_data:
            rainfall_data[key] = {'sum': 0, 'count': 0}
        rainfall_data[key]['sum'] += value
        rainfall_data[key]['count'] += 1

# Prepare the result for each Year-Month
average_rainfall = []
for (year, month), data in rainfall_data.items():
    avg_rainfall = round(data['sum'] / data['count'], 2) if data['count'] > 0 else 0
    average_rainfall.append([year, month, avg_rainfall])

# Convert the result to a DataFrame and save it to a new CSV file
df_average_rainfall = pd.DataFrame(average_rainfall, columns=['Year', 'Month', 'Average Rainfall'])
df_average_rainfall.to_csv('rainfall_reformatted.csv', index=False)

print("Reformatted file 'rainfall_reformatted.csv' has been created successfully.")
