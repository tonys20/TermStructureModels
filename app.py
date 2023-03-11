import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
import datetime
import pandas as pd
import scipy.optimize as opt

np.random.seed(4)

# Define the CIR and Vasicek models for simulating interest rates
def cir_neg(r0, K, theta, sigma, T, N):
    dt = float(T) / N
    x = np.zeros(N + 1)
    x[0] = r0
    for i in range(1, N + 1):

        dxt = K * (theta - x[i - 1]) * dt + sigma * np.sqrt(np.abs(x[i - 1])) * np.random.normal()
        x[i] = x[i - 1] + dxt

    return np.arange(0, N + 1) * dt, x

def cir(r0, K, theta, sigma, T, N):
    dt = float(T) / N
    x = np.zeros(N + 1)
    x[0] = r0
    for i in range(1, N + 1):
        if x[i-1] >=0:
            dxt = K * (theta - x[i - 1]) * dt + sigma * np.sqrt(x[i - 1]) * np.random.normal()
            x[i] = x[i - 1] + dxt
        else:
            x[i] = x[i-1]

    return np.arange(0, N + 1) * dt, x



def vasicek(r0, K, theta, sigma, T, N):
    dt = float(T) / N
    x = np.zeros(N + 1)
    x[0] = r0
    for i in range(1, N + 1):
        dxt = K * (theta - x[i - 1]) * dt + sigma * np.random.normal()
        x[i] = x[i - 1] + dxt
    return np.arange(0, N + 1) * dt, x

# Define the simulation parameters

r0 = st.slider("r0: Initial Rate", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
K = st.slider("K: Mean Reversion Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
theta = st.slider("theta: Long-term Mean", min_value=0.0, max_value=20.0, value=0.05, step=0.01)
sigma = st.slider("sigma: Volatility", min_value=0.0, max_value=5.0, value=0.1, step=0.01)
T = st.slider("T: Time to Maturity (Years)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
N = st.slider("N: Number of Time Steps", min_value=1, max_value=3650, value=10, step=1)


# Simulate interest rates using the CIR and Vasicek models
cir_x, cir_y = cir(r0, K, theta, sigma, T, N)
vasicek_x, vasicek_y = vasicek(r0, K, theta, sigma, T, N)
cir_neg_x, cir_neg_y = cir_neg(r0, K, theta, sigma, T, N)

fig = make_subplots(rows=3, cols=1, subplot_titles=("CIR Model",'CIR ABS', "Vasicek Model"))

# Add the CIR model data to the figure
fig.add_trace(
    go.Scatter(x=cir_x, y=cir_y, name="CIR Model"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=cir_neg_x, y=cir_neg_y, name = 'CIR abs hash'),
    row=2, col=1
)

# Add the Vasicek model data to the figure
fig.add_trace(
    go.Scatter(x=vasicek_x, y=vasicek_y, name="Vasicek Model"),
    row=3, col=1
)


# Update the layout of the figure
fig.update_layout(height=600, width=800, title="Modern Term Structure Models")

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)

start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2023, 3, 9)

# Get 3-month Treasury Bill data from FRED
tbill_data = web.DataReader(["DTB4WK","DTB3","DTB6","DTB1YR"], "fred", start_date, end_date).dropna()
st.table(tbill_data.head())

hist_stats = pd.DataFrame(columns =['mean', 'vol'], index = list(tbill_data.columns))
hist_stats.index = tbill_data.columns
for col in tbill_data.columns:
    hist_stats.loc[col,'mean'] = tbill_data[col].mean()
    hist_stats.loc[col,'vol'] = tbill_data[col].std()

st.table(hist_stats)


# Define the CIR model function
def CIR(r, a, b, sigma):
    dr = a*(b - r)
    dr -= 0.5*sigma**2*b**2*a*(1-np.exp(-2*r/b))/b
    dr += sigma*b*np.sqrt((1-np.exp(-2*r/b))/(2*a*b))
    return dr

# Define the error function to be minimized
def error_function(params, r):
    a, b, sigma = params
    n = len(r)
    dt = 1/252
    sum_of_errors = 0
    for i in range(1, n):
        predicted_r = r[i-1] + CIR(r[i-1], a, b, sigma)*dt
        error = r[i] - predicted_r
        sum_of_errors += error**2
    return sum_of_errors

# Load the historical data

r = tbill_data['DTB3']

# Set the initial guess for the parameters
initial_guess = [0.01, 0.01, 0.01]

# Set bounds for the parameters
bounds = [(0, None), (0, None), (0, None)]

# Optimize the parameters using the error function and initial guess
result = opt.minimize(error_function, initial_guess, args=(r,), bounds=bounds)

# Print the optimized parameters
st.write('Optimized Parameters:')
st.write(f'a = {result.x[0]}')
st.write(f'b = {result.x[1]}')
st.write(f'sigma = {result.x[2]}')