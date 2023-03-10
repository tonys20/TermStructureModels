import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def cir(r0, K, theta, sigma, T=1.0, N=10, seed=None):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + sigma*np.sqrt(rates[-1])*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

def vasicek(r0, K, theta, sigma, T=1.0, N=10, seed=None):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + sigma*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

st.title("Interest Rate Simulation Dashboard")

r0 = st.slider("Initial Rate: r_0", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
K = st.slider("Mean Reversion Rate: K", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
theta = st.slider("Long-term Mean: theta", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma = st.slider("Volatility: sigma", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
T = st.slider("Time to Maturity (Years): T", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
N = st.slider("Number of Time Steps: N", min_value=1, max_value=100, value=10, step=1)

cir_x, cir_y = cir(r0=r0, K=K, theta=theta, sigma=sigma, T=T, N=N)
vasicek_x, vasicek_y = vasicek(r0=r0, K=K, theta=theta, sigma=sigma, T=T, N=N)

# Plot the results using Plotly
fig = make_subplots(rows=2, cols=1, subplot_titles=("CIR Model", "Vasicek Model"))

fig.add_trace(
    go.Lines(x=cir_x, y=cir_y, name="CIR Model"),
    row=1, col=1
)

fig.add_trace(
    go.Lines(x=vasicek_x, y=vasicek_y, name="Vasicek Model"),
    row=2, col=1
)

fig.update_layout(height=600, width=800, title="Interest Rate Simulation")
st.plotly_chart(fig)