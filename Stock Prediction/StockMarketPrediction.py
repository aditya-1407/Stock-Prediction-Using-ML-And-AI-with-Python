import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to create a time series dataset
def create_time_series(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i+look_back)])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

# Load the historical stock price data
df = pd.read_csv('stock_data.csv')

# Extract the 'Close' column as the target variable
data = df['Close'].values.reshape(-1, 1)

# Normalize the data using preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Set the number of time steps to look back
look_back = 10

# Create the time series dataset
X, y = create_time_series(data, look_back=look_back)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform the predictions and the true values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Plot the predictions vs. true values
plt.figure(figsize=(10, 6))
# plt.scatter(y_test, label='True Values')
# plt.scatter(y_pred, label='Predictions')
sns.lineplot(data=y_test, label='True Values')
sns.lineplot(data=y_pred.flatten(), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.legend()
plt.show()
