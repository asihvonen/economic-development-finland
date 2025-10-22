#%%
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import plotly

#%%
regions = gpd.read_file('data/finland_regions.json')
regions['id'] = regions['id'].str.replace('FI', '', regex=False)
regions = regions.rename(columns={'id':'Region'})
regions

#%%
df = pd.read_csv('data/final_economic_data.csv', index_col=False)
df = df[df['Municipality'] == 'TOTAL']
df['Region'] = df['Region'].astype(str).apply(lambda x: f"{int(x):02d}")
df

#%%
fig = go.Figure()
#%%
fig_agri = px.choropleth(df, geojson='data/finland_regions.json', 
                    locations='Region', 
                    color='Gross value added (millions of euro), A Agriculture, forestry and fishing (01-03)',
                    color_continuous_scale="Viridis",
                    range_color=(0, 500),
                    scope="europe",
                    labels={'Gross value added (millions of euro), A Agriculture, forestry and fishing (01-03)':'Agriculture'},
                    animation_frame='Year',
                    )

fig_food = px.choropleth(df, geojson='data/finland_regions.json', 
                    locations='Region', 
                    color='Gross value added (millions of euro), I Accommodation and food service activities (55-56)',
                    color_continuous_scale="Viridis",
                    range_color=(0, 500),
                    scope="europe",
                    labels={'Gross value added (millions of euro), I Accommodation and food service activities (55-56)':'Food service'},
                    animation_frame='Year'
                    )


#%%
fig = go.Figure(data=fig_food.data, frames=fig_food.frames, layout=fig_food.layout)

food_frame = fig_food.frames

agri_frame = fig_agri.frames

#%%
for trace in fig_agri.data:
    trace.visible = False
    fig.add_trace(trace)

# %%
num_traces_food = len(fig_food.data)
num_traces_agri = len(fig_agri.data)

combined_frames = []
for food_frame, agri_frame in zip(food_frame, agri_frame):
    # Create visibility arrays for each frame
    food_visible = [True] * num_traces_food + [False] * num_traces_agri
    agri_visible = [False] * num_traces_food + [True] * num_traces_agri

    
    combined_frames.append(go.Frame(
        data=food_frame.data + agri_frame.data,
        name=food_frame.name,
        layout={"annotations": []}
    ))

fig.frames = combined_frames

# %%
fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=[
                        {"visible": [True] * num_traces_food + [False] * num_traces_agri},
                        {"frame": {"duration": 0}}
                    ],
                    label="Food",
                    method="update"
                ),
                dict(
                    args=[
                        {"visible": [False] * num_traces_food + [True] * num_traces_agri},
                        {"frame": {"duration": 0}}
                    ],
                    label="Agri",
                    method="update"
                )
            ]),
            direction="down",
            showactive=True,
            x=0.1,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )
    ]
)

fig.show()


# %%

#2 ways to make the html but the graph does not appear in both
plotly.offline.plot(fig, filename='Agri_Food.html')
fig.write_html('agri_food.html')
# %%
