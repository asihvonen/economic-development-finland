import folium
import pandas as pd
import branca.colormap as cm
import geopandas as gpd
from folium.plugins import TimeSliderChoropleth

class DisplayedMap:
    def __init__(self, geojson_path='data/finland_regions.json', data_path='data/regional_economic_data.csv'):
        # Load region geometries
        self.regions = gpd.read_file(geojson_path)
        self.regions['id'] = self.regions['id'].str.replace('FI', '', regex=False)
        self.regions = self.regions.rename(columns={'id': 'Region'})

        # Load base data
        self.data = pd.read_csv(data_path, index_col=False)
        self.data['Region'] = self.data['Region'].astype(str).apply(lambda x: f"{int(x):02d}")

    def create_time_map_with_updates(self, value_col: str, updates: dict, update_year: int = 2022, output_path: str = "../visualizations/updated_map.html"):
        """
        Create a TimeSliderChoropleth map, optionally applying region updates for a given year.

        Args:
            value_col (str): The column in the CSV containing the metric (e.g., 'GDP per capita').
            updates (dict): Optional. A dict of {Region: new_value}.
            update_year (int): Optional. The year to apply updates to (e.g., 2025).
            output_path (str): Optional. If given, saves map to this path.
        """

        df = pd.merge(self.data, self.regions, on='Region')

        # Convert year to string (TimeSlider requires string keys)
        df['Year'] = pd.to_datetime(df['Year'], format= '%Y') 
        df['Year'] = df['Year'].astype('int64') // 10**9
        df['Year'] = df['Year'].astype('U10')
        transformed_updated_year = pd.to_datetime(str(update_year), format='%Y')
        transformed_updated_year = str(transformed_updated_year.value // 10**9)
        # If updates dict is provided, apply them for the specified year
        if updates and update_year:
            for region, new_value in updates.items():
                region_str = f"{int(region):02d}"
                # if year exists, update it; else append
                mask = (df['Region'] == region_str) & (df['Year'] == transformed_updated_year)
                if mask.any():
                    df.loc[mask, value_col] = new_value
                    #print(df['Year'].unique())
                else:
                    df = pd.concat([df, pd.DataFrame({
                        'Region': [region_str],
                        'Year': [transformed_updated_year],
                        value_col: [new_value],
                        'geometry': [self.regions.loc[self.regions['Region'] == region_str, 'geometry'].values[0]]
                    })], ignore_index=True)
        
        # Build the color map
        values = df[value_col].dropna()
        #print(values)
        #print(f"Min: {values.min()}, Max: {values.max()}")
        max_col = max(values)
        min_col = min(values)
        cmap = cm.linear.YlOrRd_09.scale(min_col, max_col)
        df['color'] = values.map(cmap)
        # Build the style dict required by TimeSliderChoropleth
        style_dict = {}
        for i, region in enumerate(df['Region'].unique()):
            region_df = df[df['Region'] == region]
            style_dict[str(i)] = {
                row['Year']: {'color': row['color'], 'opacity': 0.7}
                for _, row in region_df.iterrows()
            }

        # Prepare geometries
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
        
        # Create map
        layer = folium.Map(min_zoom=4, max_bounds=True, tiles='cartodbpositron')
        TimeSliderChoropleth(
            data=region_gdf.to_json(),
            styledict=style_dict,
        ).add_to(layer)
        cmap.caption = value_col
        cmap.add_to(layer)

        if output_path:
            
            layer.save(output_path)
            print(f"Map saved to {output_path}")

        return layer
