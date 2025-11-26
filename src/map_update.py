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
from difflib import get_close_matches
class map_script:
    def __init__(self,region_id,value_being_shown):
        self.df_data = pd.read_csv('data/final_economic_data.csv', index_col=False)
        self.df_data = self.df_data[(self.df_data["Region"] != 3) & (self.df_data["Region"] != 20)] 
        self.region = int(region_id)
        #self.df_corr = pd.read_csv(f"data/correlation/sector_correlation_region_{self.region}.csv", index_col=0)
        self.all_regions = self.df_data['Region'].unique().tolist()
        self.value_being_shown_name = value_being_shown
        self.value_corr = pd.read_csv("data/correlation/income_gdp_correlation_all_regions.csv")
        self.ratio = pd.read_csv("data/region_gva_ratio_stats.csv")
    def update_map(self,most_affected_industry,change):
        # find top 5 correlated industries     
        #base_corr = self.df_corr.loc[features, most_affected_industry].abs().values
        gva_columns = [col for col in self.df_data.columns if col.startswith('Gross value added')]
        matches = get_close_matches(
            word=most_affected_industry, 
            possibilities=gva_columns, 
            n=1, 
            cutoff=0.6
        )

        if matches:
            matched_industry_column = matches[0]
            most_affected_industry = matched_industry_column
        else:
            # Fallback if no match meets the 0.6 cutoff
            print(f"Warning: No good match found for '{most_affected_industry}'. Skipping map update for this industry.")
            return
        region_features = []
        for region in self.all_regions:
            try:
                print(1)
                df = pd.read_csv(f"data/correlation/sector_correlation_region_{region}.csv", index_col=0)
                #region’s correlation values for the same top-5 features
                print(2)
                features = df[most_affected_industry].abs().sort_values(ascending=False).head(6).index.tolist()[1:]
                print(features)  
                corr_vec = df.loc[features, most_affected_industry].abs().values
                print(corr_vec)
                region_features.append((region, (features,corr_vec)))
            except Exception as e:
                print(f"Warning for region {region}: {e}")
        vc = self.value_corr.copy()
        vc['Region'] = vc['Region'].astype(str)
        vc['Sector'] = vc['Sector'].astype(str)
        #print(vc['Sector'].unique())
        sub = vc[vc['Sector'] == str(most_affected_industry)].copy()
        #print(sub)
        sub.set_index('Region', inplace=True)
        # corr_series maps region -> correlation scalar for the shown value column
        corr_series = sub[self.value_being_shown_name].astype(float)
        # apply per-region updates for YEAR = 2020 (or change mask to desired year)
        year_mask = (self.df_data['Year'] == 2022)
        year_data = self.df_data.loc[year_mask].copy()
        regions = year_data['Region'].astype(str).values

        # get correlation per region, default to 1.0 when missing
        corr_for_regions = corr_series.reindex(regions).fillna(1.0).values
        print(corr_for_regions)
        vr = self.ratio.copy()
        vr['Region'] = vr['Region'].astype(str)
        vr.set_index('Region', inplace=True)
        
        multipliers = np.zeros(len(regions))

        for idx, (region ,(features,corr_vec)) in enumerate(region_features):
            region_total = 0
            if(self.value_being_shown_name == 'Disposable income, net'):
                region_total = change
            else:
                main_industry_ratio_col = f"{most_affected_industry}_to_GDP_ratio_mean"
                main_industry_ratio = vr.loc[str(region), main_industry_ratio_col] if main_industry_ratio_col in vr.columns and str(region) in vr.index else 0
                print(main_industry_ratio)
                # The change applied directly to the most affected industry
                region_total += change * main_industry_ratio
                for i, industry in enumerate(features):
                    # Get correlation
                    corr = corr_vec[i]
                    
                    # Get this industry's share in this region
                    ratio_col = f"{industry}_to_GDP_ratio_mean"
                    industry_ratio = vr.loc[str(region), ratio_col] if ratio_col in vr.columns and str(region) in vr.index else 0
                    # Add contribution
                    region_total += change * corr * industry_ratio
            
            multipliers[idx] = region_total * corr_for_regions[idx]
                
        original_vals = year_data[self.value_being_shown_name].astype(float).values
        new_vals = original_vals * (1 + multipliers)

        print("\n=== UPDATE SUMMARY ===")
        print(f"Most Affected Industry: {most_affected_industry}")
        print(f"Change: {change*100:.2f}%")
        print(f"\nRegional {self.value_being_shown_name} Changes:")
        for idx, region in enumerate(regions):
            print(f"  Region {region}: {original_vals[idx]:.2f} -> {new_vals[idx]:.2f} (Change: {multipliers[idx]*100:.2f}%)")
        print("=" * 50 + "\n")
        # assign updated values back into df_data
        self.df_data.loc[year_mask, self.value_being_shown_name] = new_vals

        # prepare updates dict for map rendering
        year_data_updated = self.df_data[self.df_data['Year'] == 2022]
        updates = dict(zip(year_data_updated['Region'].astype(str), year_data_updated[self.value_being_shown_name]))
        
        #somehow store updated data
        display_map = DisplayedMap()
        display_map.create_time_map_with_updates(value_col=self.value_being_shown_name, updates=updates, update_year = 2020,output_path = "visualizations/updated_map.html")