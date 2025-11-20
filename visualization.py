import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_k_means_pca(economic_factors: pd.DataFrame):
    """
    Plot the results of k-means clustering in PCA space.
    Args:
        economic_factors : pd.DataFrame
            Dataframe which contains the columns:
            - The selected economic factors, scaled
            - Region
            - Year
            - PCA1
            - PCA2
            - Cluster
    """
    fig = px.scatter(
        economic_factors,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=["PCA1", "PCA2", "Region", "Year"],
        title="K-Means Clustering Results in PCA Space"
    )
    fig.show()


def plot_economic_factor(df, factor):
    """
    Plots an interactive time series line chart, with lines for each region,
    for the given economic factor.
    
    Args:
        df: pandas.DataFrame
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


def matplotlib_spider_chart(categories: list, values: list[list], labels: list):
    """
    Args:
        categories is a list of the variables which are used to label each value in the spider chart
        values is a list of lists, the nested lists are the numerical values used to plot the spider chart
        labels is the legend of the graph which shows what each plot instance is supposed to represent

    BE AWARE, for each nested value you HAVE to append the first element to the end of the list 
    or else the code won't work.
    """
    
    label_placement = np.linspace(start=0, stop=2*np.pi,num=len(values), endpoint=False).tolist()
    label_placement.append(label_placement[0])
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    
    for i in range(0,len(values)):
        ax.plot(label_placement,values[i],label = labels[i])

    ax.set_xticks(label_placement[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.show()


def plotly_spider_chart(categories: list, values: list[list], labels: list):
    """
    Args:
        categories is a list of the variables which are used to label each value in the spider chart
        values is a list of lists, the nested lists are the numerical values used to plot the spider chart
        labels is the legend of the graph which shows what each plot instance is supposed to represent

    BE AWARE, for each nested value you HAVE to append the first element to the end of the list 
    or else the code won't work.
    """
    fig = go.Figure()

    closed_categories = categories + [categories[0]]

    for i in range(len(values)):
        closed_values = values[i] + [values[i][0]]
        fig.add_trace(go.Scatterpolar(
            r=closed_values,
            theta=closed_categories,
            mode='lines+markers',
            name=labels[i]
        ))

    fig.update_layout(
        width=800,     
        height=800,
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=True
    )

    fig.show()
