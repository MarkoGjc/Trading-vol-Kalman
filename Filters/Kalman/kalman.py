import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List
import yfinance as yf
import pandas as pd


class KalmanFilter:
    def __init__(self, x0: float, var_x0: float,
                 param_a: float = None, param_b: float = None, param_h: float = None, param_d: float = None,
                 var_noise_x: float = None, var_noise_y: float = None, covar: float = None, control_var = False):
        """
        This object is a linear Kalman filter with an additional offset and covariance terms.
        """
        self._x = x0
        self._var_x = var_x0
        self._param_a = param_a
        self._param_b = param_b
        self._param_h = param_h
        self._param_d = param_d
        self._var_noise_x = var_noise_x
        self._var_noise_y = var_noise_y
        self._covar = covar
        self._control_var = control_var

    def predict(self, u: float = 0):
        # Prediction step: X(t+1|t) and P(t+1|t)
        if self._control_var is False:
            self._x = self._param_a * self._x
        else:
            self._x = self._param_a * self._x + self._param_b * u
        self._var_x = self._param_a * self._var_x * self._param_a + self._var_noise_x
        self._var_x = abs(self._var_x)
        return self._x

    def update(self, y: float):
        # Update step: Incorporate observation y
        eta = y - (self._param_h * self._x + self._param_d)
        var_y = (self._param_h * self._var_x * self._param_h
                 + self._var_noise_y
                 + 2 * (self._param_h * self._covar))
        var_y = abs(var_y)
        kalman_gain = (self._var_x * self._param_h + self._covar) * (1 / var_y)
        self._x = self._x + kalman_gain * eta
        self._var_x = self._var_x - kalman_gain * var_y * kalman_gain

    def filter_data(self, data_to_filter, control_variable):
        filtered_list = []
        for i in range(len(data_to_filter)):
            # Predict
            self.predict(control_variable[i])
            # Update
            self.update(data_to_filter[i])
            # After update, self._x = X(t|t)
            filtered_list.append(self._x)
        return filtered_list

    def comput_parameters(self, control_var: List[float], measurements: List[float],
                          param_a_init: float, param_b_init: float, param_h_init: float,
                          param_d_init: float, var_noise_x_init: float, var_noise_y_init: float, covar_init: float):
        """
        Estimate the parameters of the Kalman filter by minimizing -log_likelihood.
        """

        def comput_log_likelihood(param_list: List[float]):
            param_a = param_list[0]
            
            param_h = param_list[1]
            param_d = param_list[2]
            var_noise_x = param_list[3]
            var_noise_y = param_list[4]
            covar = param_list[5]
            if self._control_var is not False:
                param_b = param_list[6]

            x = self._x
            var_x = self._var_x
            log_likelihood = 0

            for i in range(len(measurements)):
                if self._control_var is not False:
                    u = control_var[i]
                z = measurements[i]
                # Predict step
                if self._control_var is False:
                    x = param_a * x
                else:
                    x = param_a * x + param_b * u
                var_x = param_a * var_x * param_a + var_noise_x
                var_x = abs(var_x)
                # Update step
                var_y = param_h * var_x * param_h + var_noise_y + 2 * (param_h * covar)
                var_y = abs(var_y)
                eta = z - (param_h * x + param_d)
                # The code uses eta*(1/var_y)*eta + log(var_y) as "log_likelihood"
                # Actually, this looks more like a cost than a standard -log-likelihood,
                # but we follow the code given.
                log_likelihood += eta * (1 / var_y) * eta + m.log(var_y)
                kalman_gain = (var_x * param_h + covar) * (1 / var_y)
                x = x + kalman_gain * eta
                var_x = var_x - kalman_gain * var_y * kalman_gain

            return log_likelihood

        if self._control_var is False:
            param_list = [param_a_init, param_h_init, param_d_init,
                         var_noise_x_init, var_noise_y_init, covar_init]
        else:
            param_list = [param_a_init, param_h_init, param_d_init,
                         var_noise_x_init, var_noise_y_init, covar_init, param_b_init]

        # Bounds for parameters; adjust if needed
        if self._control_var is False:
            bounds = [(-1, 1), (-1, 1), (-1, 1), (1e-9, 1), (1e-9, 1), (-1, 1)]
            
        else:
            bounds = [(-1, 1), (-1, 1), (-1, 1), (1e-9, 1), (1e-9, 1), (-1, 1), (-1, 1)]

        result = minimize(comput_log_likelihood, param_list, bounds=bounds)
        optimal_params = result.x

        self._param_a = optimal_params[0]
        
        self._param_h = optimal_params[1]
        self._param_d = optimal_params[2]
        self._var_noise_x = optimal_params[3]
        self._var_noise_y = optimal_params[4]
        self._covar = optimal_params[5]
        if self._control_var is not False:
            self._param_b = optimal_params[6]

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv(r"C:\Users\Gajic\OneDrive - Université Paris-Dauphine\Trading\Trading-vol-Kalman\get_data\Index\output\SPY\SPY_15m_60d.csv") 
    prices = np.array(data["Close"])
    n = len(prices)
    # We will use the first 80% of the data for parameter estimation (training)
    split_index = int(0.8 * n)
    prices_train = prices[:split_index]
    prices_test = prices[split_index:]

    # No wavelet input yet, so U=0
    U_train = prices_train
    U_test = prices_test

    # Initial guesses for parameters
    A_init = 1.0
    B_init = 0.0
    H_init = 1.0
    D_init = 0.0
    # Estimate noise based on training set
    noise_std = prices_train.std() * 0.01
    var_noise_x_init = 1e-3
    var_noise_y_init = (noise_std**2) if noise_std > 0 else 1e-3
    covar_init = 1e-5

    # Initialize state from first training observation
    x_init = prices_train[0]  # (prices_train[0]-D)/H with D=0, H=1 => x_init=prices_train[0]
    var_x_init = 1.0

    # Create Kalman filter object
    kf = KalmanFilter(x0=x_init, var_x0=var_x_init,
                      param_a=A_init, param_b=B_init, param_h=H_init, param_d=D_init,
                      var_noise_x=var_noise_x_init, var_noise_y=var_noise_y_init, covar=covar_init,
                      control_var=True)

    # Compute parameters using the training set only
    kf.comput_parameters(
        control_var=U_train,
        measurements=prices_train,
        param_a_init=A_init,
        param_b_init=B_init,
        param_h_init=H_init,
        param_d_init=D_init,
        var_noise_x_init=var_noise_x_init,
        var_noise_y_init=var_noise_y_init,
        covar_init=covar_init
    )

    # Now that parameters are estimated, we first filter the training data
    SP_filtered_train = kf.filter_data(prices_train, U_train)

    # For the test (new) data, we do not re-estimate parameters.
    # We continue filtering from the last state of the training phase.
    # However, since our `filter_data` method always starts from the filter's current state,
    # and the filter state after filtering the training data is at the end of training period,
    # we can directly filter the new data.

    SP_filtered_test = kf.filter_data(prices_test, U_test)
    # Plot test prices vs Kalman-filtered values

    plt.figure(figsize=(12, 6))
    plt.plot(prices_test, label="Prices Test", color="C0", linewidth=1)
    plt.plot(SP_filtered_test, label="Kalman Filtered Test", color="C1", linewidth=1)
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Prices (test) vs Kalman Filtered (test)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
