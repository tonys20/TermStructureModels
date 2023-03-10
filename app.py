import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Define the CIR and Vasicek models for simulating interest rates
def cir(r0, K, theta, sigma, T, N):
    dt = float(T) / N
    x = np.zeros(N + 1)
    x[0] = r0
    for i in range(1, N + 1):
        dxt = K * (theta - x[i - 1]) * dt + sigma * np.sqrt(x[i - 1]) * np.random.normal()
        x[i] = x[i - 1] + dxt
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

r0 = st.slider("Initial Rate", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
K = st.slider("Mean Reversion Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
theta = st.slider("Long-term Mean", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma = st.slider("Volatility", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
T = st.slider("Time to Maturity (Years)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
N = st.slider("Number of Time Steps", min_value=1, max_value=100, value=10, step=1)


# Simulate interest rates using the CIR and Vasicek models
cir_x, cir_y = cir(r0, K, theta, sigma, T, N)
vasicek_x, vasicek_y = vasicek(r0, K, theta, sigma, T, N)

# Create a new Plotly figure
fig = make_subplots(rows=2, cols=1, subplot_titles=("CIR Model", "Vasicek Model"))

# Add the CIR model data to the figure
fig.add_trace(
    go.Scatter(x=cir_x, y=cir_y, name="CIR Model"),
    row=1, col=1
)

# Add the Vasicek model data to the figure
fig.add_trace(
    go.Scatter(x=vasicek_x, y=vasicek_y, name="Vasicek Model"),
    row=2, col=1
)

# Update the layout of the figure
fig.update_layout(height=600, width=800, title="Interest Rate Simulation")

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)
