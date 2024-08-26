import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#load dataset
df = pd.read_csv('C:\\Users\\Vartika.Yadav\\Desktop\\ai-ml\\weather.csv')

df['date'] = pd.to_datetime(df['date'])

#Extract the day of the year from the 'date' column to use as a feature
df['day_of_year'] = df['date'].dt.dayofyear

#Handle categorical data in the 'weather' column by applying one-hot encoding
df = pd.get_dummies(df, columns=['weather'], drop_first=True)

#Select features for prediction
#'day_of_year', 'precipitation', 'wind', and one-hot encoded weather columns
X = df[['day_of_year', 'precipitation', 'wind'] + [col for col in df.columns if col.startswith('weather_')]]

#Define target variables (temp_max and temp_min)
y_max = df['temp_max']
y_min = df['temp_min']

#Split the data into training and testing sets
X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(X, y_max, test_size=0.2, random_state=42)
X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=42)

#Create and train the linear regression model for temp_max
model_max = LinearRegression()
model_max.fit(X_train_max, y_train_max)

#Create and train the linear regression model for temp_min
model_min = LinearRegression()
model_min.fit(X_train_min, y_train_min)

#Predict max temperature using the test set and calculate Mean Squared Error
y_pred_max = model_max.predict(X_test_max)
mse_max = mean_squared_error(y_test_max, y_pred_max)
print(f"Mean Squared Error for max temperature prediction: {mse_max}")

#Predict min temperature using the test set and calculate Mean Squared Error
y_pred_min = model_min.predict(X_test_min)
mse_min = mean_squared_error(y_test_min, y_pred_min)
print(f"Mean Squared Error for min temperature prediction: {mse_min}")

#Visualize the results - Actual vs Predicted Max Temperature
plt.scatter(y_test_max, y_pred_max, color='red')
plt.xlabel('Actual Max Temperature')
plt.ylabel('Predicted Max Temperature')
plt.title('Max Temperature Prediction')
plt.show()

#Visualize the results - Actual vs Predicted Min Temperature
plt.scatter(y_test_min, y_pred_min, color='blue')
plt.xlabel('Actual Min Temperature')
plt.ylabel('Predicted Min Temperature')
plt.title('Min Temperature Prediction')
plt.show()

#Predict max and min temperature for a new day (example)
# Replace with actual values for day_of_year, precipitation, wind, and weather conditions
new_data = pd.DataFrame({
    'day_of_year': [210],  # Example: 210th day of the year
    'precipitation': [0.5],  # Example: 0.5 mm
    'wind': [3.2],  # Example: 3.2 m/s
    'weather_sunny': [1],  # Example: sunny day
    'weather_rainy': [0],
})

# Ensure new_data has the same columns as X_train
new_data = new_data.reindex(columns=X_train_max.columns, fill_value=0)

# Predict max and min temperatures for the new day
predicted_max_temp = model_max.predict(new_data)
predicted_min_temp = model_min.predict(new_data)

print(f"Predicted max temperature: {predicted_max_temp[0]:.2f}°C")
print(f"Predicted min temperature: {predicted_min_temp[0]:.2f}°C")

#Create a DataFrame to compare actual and predicted values
comparison_df = pd.DataFrame({
    'Actual Max Temp': y_test_max,
    'Predicted Max Temp': y_pred_max,
    'Actual Min Temp': y_test_min,
    'Predicted Min Temp': y_pred_min
})

#Reset index for better readability
comparison_df = comparison_df.reset_index(drop=True)

#Display the comparison
print("Side-by-side comparison of actual vs. predicted temperatures:")
print(comparison_df)

#Visualize the comparison for max temperature
plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df['Actual Max Temp'], label='Actual Max Temp', color='blue', marker='o')
plt.plot(comparison_df.index, comparison_df['Predicted Max Temp'], label='Predicted Max Temp', color='red', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('Max Temperature')
plt.title('Actual vs Predicted Max Temperature')
plt.legend()
plt.show()

#Visualize the comparison for min temperature
plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df['Actual Min Temp'], label='Actual Min Temp', color='blue', marker='o')
plt.plot(comparison_df.index, comparison_df['Predicted Min Temp'], label='Predicted Min Temp', color='red', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('Min Temperature')
plt.title('Actual vs Predicted Min Temperature')
plt.legend()
plt.show()