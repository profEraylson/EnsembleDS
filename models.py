from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE


def train_linear_regression(x_train, y_train, x_val, y_val):

    modelo = LinearRegression() 
    modelo.fit(x_train, y_train)
    prev_v = modelo.predict(x_val) 
    mse_val  = MSE(y_val, prev_v)
 

    return modelo, mse_val