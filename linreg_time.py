#%%
import numpy as np
import pandas as pd 
from xgboost import XGBRegressor
import plotly.graph_objects as go
import sklearn
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression, Ridge
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from sklearn.metrics import mean_absolute_error

#%%
df = pd.read_csv('data/finland_gdp_by_quarter_(large).csv', index_col=False)
df['observation_date'] = pd.to_datetime(df['observation_date'], format="%Y-%m-%d")
df = df.set_index('observation_date')
df.index = pd.DatetimeIndex(df.index, freq='QS')

#%%
df_train = df.iloc[:81]
df_test = df.iloc[121:133] 
df_validate = df.iloc[133:]

# %%
# Create forecaster
# ==============================================================================
window_features = RollingFeatures(stats=["mean"], window_sizes=[4])
forecaster = ForecasterRecursive(
                regressor       = LinearRegression(),#XGBRegressor(random_state=15, enable_categorical=True),
                lags            = [1, 2, 3, 4, 8],
                window_features = window_features,
             )

# Train forecaster
# ==============================================================================
y = df_train['CLVMNACSCAB1GQFI']
forecaster.fit(y=y)

#%%
# Backtest model on test data
# ==============================================================================
cv = TimeSeriesFold(
        steps              = 4,
        initial_train_size = len(df_train),
        refit              = False,
)
metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = df['CLVMNACSCAB1GQFI'],
                            cv         = cv,
                            metric     = 'mean_absolute_error'
                       )

#%%
fig = go.Figure()
fig.add_trace(go.Scatter(x=predictions.index, y=predictions.pred, mode='lines', name='Pred'))
fig.add_trace(go.Scatter(x=df.index, y=df['CLVMNACSCAB1GQFI'], mode='lines', name='Real'))
fig.update_layout(
    title  = 'GDP',
    xaxis_title="Year",
    yaxis_title="GDP",
    legend_title="Partition:",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1, xanchor="left", x=0.001)
)
fig.show()

