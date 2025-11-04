

import folium
import pandas as pd
import branca.colormap as cm
import geopandas as gpd
from folium.plugins import TimeSliderChoropleth
import numpy as np

class dispalyed_map:   
    def __init__(self):
        self.regions = gpd.read_file('data/finland_regions.json')
        self.regions['id'] = regions['id'].str.replace('FI', '', regex=False)
        regions = regions.rename(columns={'id':'Region'})


        self.data = pd.read_csv('data/final_economic_data.csv', index_col=False)
        self.data = self.data[self.data['Municipality'] == 'TOTAL']
        self.data['Region'] = self.data['Region'].astype(str).apply(lambda x: f"{int(x):02d}")


    def time_map(value_col: str,self): 
        df = pd.merge(self.data, self.regions, on='Region')
        layer = folium.Map(min_zoom=4, max_bounds=True,tiles='cartodbpositron')

        df['Year'] = pd.to_datetime(df['Year'], format= '%Y') 
        df['Year'] = df['Year'].astype('int64') // 10**9
        df['Year'] = df['Year'].astype('U10')

        values = df[value_col].dropna()

        max_col = max(values)
        min_col = min(values)
        cmap = cm.linear.YlOrRd_09.scale(min_col, max_col)
        df['color'] = values.map(cmap)

        region_list = df['Region'].unique().tolist()
        region_idx = range(len(region_list))

        style_dict = {}

        for i in region_idx:
            region = region_list[i]
            res = df[df['Region'] == region]
            inner = {}
            for _, r in res.iterrows():
                inner[r['Year']] = {'color': r['color'], 'opacity': 0.7}
                style_dict[str(i)] = inner

        region_df = df[['geometry']]
        region_gdf = gpd.GeoDataFrame(region_df)
        region_gdf = region_gdf.drop_duplicates().reset_index()


        TimeSliderChoropleth(
            data=region_gdf.to_json(),  
            styledict=style_dict,
        ).add_to(layer)

        cmap.add_to(layer)

        return layer

    # map = time_map('GDP per capita (euro at current prices)')
    # map.save('gdp_map.html')

