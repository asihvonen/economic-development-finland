#%%
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import importlib
import visualization
importlib.reload(visualization)
import numpy as np



#%%
region = st.query_params["region"]
year = 2023
df = pd.read_csv("data/final_economic_data.csv", index_col=False)
df_radar = df[df['Year'] == year-1]


#%%
df = df[df['Region'] == int(region)]
df_year = df[df['Year'] == year]
df_lastyear = df[df['Year'] == year - 1]

#%%
df_reg = pd.read_csv("data/region_description.csv", index_col=False)

#%%

industries = list(filter(lambda x: 'Gross value added (millions of euro)' in x, df.columns))
col_short_map_compact = {
    "Gross value added (millions of euro), C Manufacturing (10-33)": "Manufacturing (C)",
    "Gross value added (millions of euro), D, E Electricity, gas, steam and air conditioning and water supply; sewerage and waste management (35-39)": "Energy & Water (D–E)",
    "Gross value added (millions of euro), L Real estate activities": "Real Estate (L)",
    "Gross value added (millions of euro), A Agriculture, forestry and fishing (01-03)": "Agriculture (A)",
    "Gross value added (millions of euro), B Mining and quarrying (05-09)": "Mining (B)",
    "Gross value added (millions of euro), F Construction (41-43)": "Construction (F)",
    "Gross value added (millions of euro), G Wholesale and retail trade; repair of motor vehicles and motorcycles (45-47)": "Trade & Vehicle Repair (G)",
    "Gross value added (millions of euro), H Transportation and storage (49-53)": "Transport & Storage (H)",
    "Gross value added (millions of euro), I Accommodation and food service activities (55-56)": "Accommodation & Food (I)",
    "Gross value added (millions of euro), J Information and communication (58-63)": "Info & Communication (J)",
    "Gross value added (millions of euro), K Financial and insurance activities (64-66)": "Finance & Insurance (K)",
    "Gross value added (millions of euro), M Professional, scientific and technical activities (69-75)": "Professional Services (M)",
    "Gross value added (millions of euro), N Administrative and support service activities (77-82)": "Admin & Support (N)",
    "Gross value added (millions of euro), O Public administration and defence; compulsory social security (84)": "Public Admin & Defence (O)",
    "Gross value added (millions of euro), P Education (85)": "Education (P)",
    "Gross value added (millions of euro), Q Human health and social work activities (86-88)": "Health & Social Work (Q)",
    "Gross value added (millions of euro), R, S Other service activities (90-96)": "Other Services (R–S)",
    "Gross value added (millions of euro),  T Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use(97-98)": "Household production (T)",
}


#%%

def calc_delta(column: str) -> float:
    delta = ((df_year[column].values - df_lastyear[column].values)/df_lastyear[column].values)[0]
    return round(delta, 3)



st.set_page_config(layout="wide")
#%%
with st.sidebar:
    st.markdown(f"<h1 style='font-size: 60px;'>Region: {df_reg[df_reg['Region'] == int(region)]['Name'].values[0]}</h1>", unsafe_allow_html=True)
    st.markdown(df_reg[df_reg['Region'] == int(region)]['Description'].values[0])
    st.metric(label='Population', value=df_year['Total (population)'], border=True, delta=calc_delta('Total (population)'))
    st.metric(label='Unemployed jobseekers',value=df_year['Unemployed jobseekers'], border=True, delta=calc_delta('Unemployed jobseekers'), delta_color="inverse")
    st.metric(label='Imports', value=df_year['Imports (euro)'], border=True, delta=calc_delta('Imports (euro)'))
    st.metric(label='Exports', value=df_year['Exports (euro)'], border=True, delta=calc_delta('Exports (euro)'))


#first row of graph
col = st.columns((5, 5, 5), gap='medium', border=True)

with col[0]:
    fig1 = px.line(df, x="Year", y="GDP per capita (euro at current prices)", title="GDP over time")
    st.plotly_chart(fig1, use_container_width=True)

with col[1]:
    fig2 = px.line(df, x="Year", y="Disposable income, net", title="Disposable income")
    st.plotly_chart(fig2, use_container_width=True)

with col[2]:
    data = df[df['Year'] >= 2015]
    fig3 = px.line(data, x="Year", y=['Imports (euro)', 'Exports (euro)'], title="Import and Exports")
    st.plotly_chart(fig3, use_container_width=True)


#second row of graph
with st.container(border=True):

    st.session_state['show_line_graph'] = True


    selected_industries = st.multiselect('Select an Industry', options=industries, default=industries[0], key='line')


    if st.session_state.show_line_graph:
        fig4 = px.line(df, x='Year', y=selected_industries, title='Industries')
        st.plotly_chart(fig4, use_container_width=True)




#3rd row of graph
col3 = st.columns((5, 5), gap='medium', border=True)

with col3[0]:
    poplulation = df_year[df_year.columns[df_year.columns.str.contains('Total \(population\)', regex=True)]]
    poplulation = poplulation.drop('Total (population)', axis=1)
    values = poplulation.iloc[0].values
    fig5 = go.Figure(data=[go.Bar(x=poplulation.columns, y=values)])
    fig5.update_layout(
        title='Population by Age Group',
        xaxis_title='Age Group',
        yaxis_title='Population',
        xaxis_tickangle=-45  # Rotate labels for better readability
    )
    st.plotly_chart(fig5, use_container_width=True)

with col3[1]: 

    regions = df_radar['Region'].unique()
    st.session_state['show_radar'] = True
    

    selected_industry = st.multiselect('Select an Industry', options=industries, default=industries[0], key='radar')
    selected_regions =  st.multiselect('Select a region', options=regions, default=int(region))
    values = df_radar[df_radar['Region'].isin(selected_regions)][selected_industry].values.tolist()
    industry_short = list(map(lambda x: col_short_map_compact[x], selected_industry))



    if st.session_state.show_radar:
        fig = visualization.plotly_radar(selected_regions, values, industry_short)
        st.plotly_chart(fig, use_container_width=True)


