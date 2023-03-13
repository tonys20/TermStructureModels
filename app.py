import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
import datetime
import pandas as pd
import scipy.optimize as opt
import base64



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
        if x[i-1] >0:
            dxt = K * (theta - x[i - 1]) * dt + sigma * np.sqrt(x[i - 1]) * np.random.normal()
        else:
            dxt = K * (theta - x[i - 1]) * dt
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


def get_tbill_data(item_ls, start_date, end_date):
    tbill_data = web.DataReader(item_ls, "fred", start_date, end_date).dropna()
    return tbill_data

#get the t bill data from FRED
items = ["DTB4WK","DTB3","DTB6","DTB1YR"]
start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime.today()

tbill_data = get_tbill_data(items, start_date, end_date)



hist_stats = pd.DataFrame(columns =['mean', 'vol'], index = list(tbill_data.columns))
hist_stats.index = tbill_data.columns
for col in tbill_data.columns:
    hist_stats.loc[col,'mean'] = tbill_data[col].mean()
    hist_stats.loc[col,'vol'] = tbill_data[col].std()

left_col, right_col = st.columns(2)
with left_col:
    '''
    # Modern Interest Rate Models Dashboard
    This section demonstrates effect of changing each of the parameters in the 3 models
    '''
    options = ["DTB4WK","DTB3","DTB6","DTB1YR"]
    rate_selected = st.selectbox('Select the Treasury Bill rate to calibrate to', options)
    r0_cal = 0.5
    theta_cal = 0.0
    sigma_cal = 0.0
    N_cal = 12.0
    T_cal = 10.0
    if st.button('Calibrate for r0, LT mean, volatility!'):
        r0_cal = tbill_data[rate_selected].iloc[-1]
        theta_cal = hist_stats.loc[rate_selected,'mean']
        sigma_cal = hist_stats.loc[rate_selected,'vol']

    N = st.number_input("N: Number of Time Steps", min_value=1, max_value=2000, value=12 , step=10)

    # Define the simulation parameters

    T = st.number_input("T: Time to Maturity (Years)", min_value=0, max_value=10, value=10 , step=1)
    
    K = st.slider("K: Mean Reversion Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    r0 = st.slider("r0: Initial Rate", min_value=0.0, max_value=6.0, value=float(r0_cal), step=0.01)
    theta = st.slider("theta: Long-term Mean", min_value=0.0, max_value=6.0, value=float(theta_cal), step=0.01)
    sigma = st.slider("sigma: Volatility", min_value=0.0, max_value=3.0, value=float(sigma_cal), step=0.01)
    
    



    # Simulate interest rates using the CIR and Vasicek models
    cir_x, cir_y = cir(r0, K, theta, sigma, T, N)
    vasicek_x, vasicek_y = vasicek(r0, K, theta, sigma, T, N)
    cir_neg_x, cir_neg_y = cir_neg(r0, K, theta, sigma, T, N)
    data = {
        'x': cir_x,
        'cir_y': cir_y,
        'vascicek_y':vasicek_y,
        'cir_neg_y':cir_neg_y
    }
    output_df = pd.DataFrame(data)
    csv = output_df.to_csv(index=False)

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
    if st.button('Download data'):
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

with right_col:
    st.table(tbill_data.tail(3))
    st.table(hist_stats)


# Define the CIR model function
def cir_opt(r0, K, theta, sigma, T, N):
    dt = float(T) / N
    x = np.zeros(N + 1)
    x[0] = r0
    for i in range(1, N + 1):
        if x[i-1] >=0:
            dxt = K * (theta - x[i - 1]) * dt + sigma * np.sqrt(x[i - 1]) * np.random.normal()
        else:
            dxt = 0
    return dxt


# Define the error function to be minimized
#def error_function(K, r):
 #   sigma = sigma_cal
  #  theta = theta_cal
   # n = len(r)
    #dt = N/T
    #sum_of_errors = 0
    #for i in range(1, n):
    #    predicted_r = r[i-1] + cir_opt(r[i-1], K, theta, sigma,n*dt,n)*dt
    #    error = r[i] - predicted_r
    #    sum_of_errors += error**2
    #return sum_of_errors


# Load the historical data

#r = tbill_data[rate_selected]
#sse1 = error_function(0.01, r)

# Set the initial guess for the parameters
#initial_guess = 0.02

# Set bounds for the parameters
#bounds = (0, 0.1)

# Optimize the parameters using the error function and initial guess

#result = opt.minimize_scalar(error_function, args=(r,), bounds=bounds, method = 'bounded')



#st.write(f'K = {float(result.x)}')