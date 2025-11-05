import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import plotly
import folium
import branca.colormap as cm
from folium.plugins import TimeSliderChoropleth
from folium_choropleth import DisplayedMap
class map_script:
    def __init__(self,region_id,value_being_shown):
        self.df_data = pd.read_csv('data/regional_economic_data.csv', index_col=False)
        self.region = int(region_id)
        self.df_corr = pd.read_csv(f"data/correlation/sector_correlation_region_{self.region}.csv", index_col=0)
        self.all_regions = self.df_data['Region'].unique().tolist()
        self.value_being_shown_name = value_being_shown
    def update_map(self,most_affected_industry,change):
        #print(f"Available regions in df: {self.all_regions}")    
        features = self.df_corr[most_affected_industry].abs().sort_values(ascending=False).head(5).index.tolist()[1:] # find top 5 correlated industries      
        #print(f"Top 5 features for industry {most_affected_industry}: {features}")
        base_corr = self.df_corr.loc[features, most_affected_industry].abs().values
        similarities = []
        for region in self.all_regions:
            try:
                df = pd.read_csv(f"data/correlation/sector_correlation_region_{region}.csv", index_col=0)
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
        #print(f"Top 3 similar regions to {self.region} for industry {most_affected_industry}: {closest}")
        #corr_with_show_value = self.df_corr.loc[:, self.value_being_shown.name]
        corr_with_show_value = 0.7  # Placeholder for actual correlation value
        
        mask_base = (self.df_data['Region'] == self.region) & (self.df_data['Year'] == 2020)
        #print(self.df_data.loc[mask_base, self.value_being_shown_name])
        print(f"Applying change of {change*100}% to region {self.region} for industry {most_affected_industry}")
        self.df_data.loc[mask_base, self.value_being_shown_name] *= (1 + change) * corr_with_show_value
        for region_id, sim_score in closest:
            mask = (self.df_data['Region'] == region_id) & (self.df_data['Year'] == 2020)
            print(f"Applying similarity-based change to region {region_id} with similarity score {sim_score}")
            self.df_data.loc[mask, self.value_being_shown_name] *= (1 + change)*corr_with_show_value * (0.5 + 0.5*sim_score)
        # Get only the target year data for updates
        year_data = self.df_data[self.df_data['Year'] == 2020]
        updates = dict(zip(year_data['Region'], year_data[self.value_being_shown_name]))
        
        #somehow store updated data
        
        display_map = DisplayedMap()
        display_map.create_time_map_with_updates(value_col=self.value_being_shown_name, updates=updates, update_year = 2020,output_path = "visualizations/updated_map.html")