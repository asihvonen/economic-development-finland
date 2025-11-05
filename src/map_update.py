import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import plotly
import folium
import branca.colormap as cm
from folium.plugins import TimeSliderChoropleth
from folium_choropleth import dispalyed_map
class map_script:
    def __init__(self,region_id,value_being_shown):
        self.df_data = pd.read_csv('data/final_economic_data.csv', index_col=False)
        self.region = int(region_id)
        self.df_corr = pd.read_csv(f"data/correlation/sector_correlation_{self.region}.csv", index_col=0)
        self.all_regions = self.df_data['Region'].unique().tolist()
        self.value_being_shown = self.df_data[value_being_shown]
    def update_map(self,most_affected_industry,change):    
        features = self.df_corr[most_affected_industry].abs().sort_values(ascending=False).head(5).index.tolist()[1:] # find top 5 correlated industries      
        base_corr = self.df_corr.loc[features, most_affected_industry].abs().values
        similarities = []
        for region in self.all_regions:
            try:
                df = pd.read_csv(f"data/correlation/sector_correlation_{region}.csv", index_col=0)
                # get that region’s correlation values for the same top-5 features
                corr_vec = df.loc[features, most_affected_industry].abs().values
                # compute cosine similarity (always between 0 and 1)
                sim = np.dot(base_corr, corr_vec) / (
                    np.linalg.norm(base_corr) * np.linalg.norm(corr_vec)
                )
                similarities.append((region, sim))
            except Exception as e:
                print(f"Warning for region {region}: {e}")
        # sort and take top 3 similar (excluding itself)
        closest = sorted(similarities, key=lambda x: x[1], reverse=True)
        closest = [(r, round(s, 3)) for r, s in closest if r != self.region][:3]
        
        #corr_with_show_value = self.df_corr.loc[:, self.value_being_shown.name]
        corr_with_show_value = 0.7  # Placeholder for actual correlation value
        
        self.df_data.loc[self.df_data['Region'] == self.region, self.value_being_shown.name] *= (1 + change)*corr_with_show_value 
        for c in closest:
            self.df_data.loc[self.df_data['Region'] == c, self.value_being_shown.name] *= (1 + corr_with_show_value * (0.5 + 0.5*similarities.loc[c]))
        
        #somehow store updated data
        
        visual_map = dispalyed_map.time_map(self.value_being_shown.name)
        
        return visual_map