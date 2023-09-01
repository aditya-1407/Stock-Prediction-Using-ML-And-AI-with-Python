# Stock Price Prediction using LSTM

This repository contains Python code for building and training a Long Short-Term Memory (LSTM) neural network model to predict stock prices. The model uses historical stock price data, preprocesses it, and then trains an LSTM network to make predictions. This README file provides an overview of the code and its functionality.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow (with Keras)

You can install these dependencies using pip:

```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Code Overview

The code consists of several steps:

1. Importing necessary libraries.
2. Defining a function to create a time series dataset.
3. Loading historical stock price data from a CSV file.
4. Preprocessing the data by normalizing it.
5. Creating a time series dataset from the normalized data.
6. Splitting the data into training and testing sets.
7. Building an LSTM model for stock price prediction.
8. Training the model on the training data.
9. Making predictions on the test data.
10. Inverse transforming the predictions and true values to their original scales.
11. Calculating the Root Mean Squared Error (RMSE) to evaluate the model's performance.
12. Plotting the predictions against the true values.

## Usage

1. Ensure that you have the necessary prerequisites installed as mentioned above.

2. Place your historical stock price data in a CSV file. Update the following line in the code to specify the correct file path:

   ```
   df = pd.read_csv('firstfilework\stock_data.csv')
   ```

3. Configure the model and training parameters according to your requirements, such as the number of LSTM units, learning rate, and the number of epochs.

4. Run the code using a Python interpreter.

   ```
   python stock_price_prediction.py
   ```

5. The code will load your data, preprocess it, train the LSTM model, make predictions, and display the RMSE along with a plot comparing the predicted and true stock prices.

## Customization

Feel free to customize the code to fit your specific needs. You can experiment with different hyperparameters, try different LSTM architectures, or use alternative evaluation metrics for model performance.

## License

This code is provided under the MIT License. You are free to use, modify, and distribute it as needed. Please refer to the LICENSE file for more details.

## Acknowledgments

- This code was created for educational and demonstration purposes.
- The data used in this example is for illustrative purposes only. In a real-world scenario, you should replace it with your own financial data.

Happy stock price prediction!
