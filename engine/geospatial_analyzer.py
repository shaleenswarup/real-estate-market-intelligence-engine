"""Geospatial Analysis Module for Real Estate Intelligence.

Handles spatial clustering, neighborhood analysis, and proximity calculations.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
import folium
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

class GeoSpatialAnalyzer:
    """Advanced geospatial analysis for real estate markets."""
    
    def __init__(self, crs='EPSG:4326'):
        self.crs = crs
        self.gdf = None
        
    def load_properties(self, df, lat_col='latitude', lon_col='longitude'):
        """Load properties into GeoDataFrame."""
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        self.gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)
        logger.info(f"Loaded {len(self.gdf)} properties")
        return self.gdf
    
    def spatial_clustering(self, eps=0.01, min_samples=5, method='dbscan'):
        """Perform spatial clustering on properties.
        
        Args:
            eps: Maximum distance between samples (in degrees)
            min_samples: Minimum points in cluster
            method: 'dbscan' or 'kmeans'
        
        Returns:
            Updated GeoDataFrame with cluster labels
        """
        coords = np.array([self.gdf.geometry.x, self.gdf.geometry.y]).T
        
        if method == 'dbscan':
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            self.gdf['cluster'] = clustering.labels_
        else:
            clustering = KMeans(n_clusters=10, random_state=42).fit(coords)
            self.gdf['cluster'] = clustering.labels_
        
        logger.info(f"Found {len(set(self.gdf['cluster'])) - 1} spatial clusters")
        return self.gdf
    
    def calculate_neighborhood_stats(self, neighborhood_radius_km=2):
        """Calculate neighborhood characteristics."""
        radius_deg = neighborhood_radius_km / 111  # Convert km to degrees
        stats = []
        
        for idx, row in self.gdf.iterrows():
            point = row.geometry
            nearby = self.gdf[self.gdf.geometry.distance(point) <= radius_deg]
            
            stats.append({
                'property_id': idx,
                'neighbor_count': len(nearby),
                'avg_price': nearby['price'].mean(),
                'median_price': nearby['price'].median(),
                'price_variance': nearby['price'].std()
            })
        
        return gpd.GeoDataFrame(stats, crs=self.crs)
    
    def create_heatmap(self, output_path='heatmap.html'):
        """Create interactive heatmap of property values."""
        center = [self.gdf.geometry.y.mean(), self.gdf.geometry.x.mean()]
        m = folium.Map(location=center, zoom_start=12)
        
        for idx, row in self.gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=f"${row['price']:,.0f}",
                color='red' if row['price'] > self.gdf['price'].quantile(0.75) else 'blue',
                fill=True
            ).add_to(m)
        
        m.save(output_path)
        logger.info(f"Heatmap saved to {output_path}")
        return m
