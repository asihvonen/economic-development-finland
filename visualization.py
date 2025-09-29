import plotly.express as px

def plot_economic_factor(df, factor):
    """
    Plots an interactive time series line chart, with lines for each region,
    for the given economic factor.
    
    Args:
        df : pandas.DataFrame
            DataFrame with columns ["Region", "Year", <economic factors>].
        factor : str
            Name of the economic factor column to plot.
    """
    fig = px.line(
        df,
        x="Year",
        y=factor,
        color="Region",
        markers=True,
        title=f"Time Series of {factor} by Region"
    )
    
    # Add hover info and legend interactivity
    fig.update_traces(hovertemplate="Year: %{x}<br>" + factor + ": %{y}<br>Region: %{legendgroup}")
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="Region",
    )
    
    fig.show()