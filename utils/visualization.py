import folium
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional
from constants import DEFAULT_CONFIG
from typing import List, Tuple

from streamlit_folium import folium_static

class MapVisualizer:
    def __init__(self):
        self.config = DEFAULT_CONFIG["visualization"]

    def create_base_map(self, locations: List[Tuple[float, float]] = None) -> folium.Map:
        """Create base map centered on locations or default"""
        if locations:
            lats = [loc[0] for loc in locations]
            lons = [loc[1] for loc in locations]
            center = [sum(lats)/len(lats), sum(lons)/len(lons)]
            zoom = self._calculate_zoom(locations)
        else:
            center = self.config["map_center"]
            zoom = self.config["default_zoom"]
        
        return folium.Map(
            location=center,
            zoom_start=zoom,
            tiles="cartodbpositron"
        )

    def create_single_location_map(self, location_data: Dict) -> folium.Map:
        """Creates a map focused on a single candidate location"""
        m = folium.Map(
            location=[location_data["lat"], location_data["lon"]],
            zoom_start=12,
            tiles="cartodbpositron"
        )
        
        # Add the candidate marker
        folium.Marker(
            [location_data["lat"], location_data["lon"]],
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
            tooltip=f"Score: {location_data.get('demand_score', 0):.2f}",
            popup=self._create_candidate_popup(location_data)
        ).add_to(m)
        
        # Add a circle showing the competition radius
        folium.Circle(
            location=[location_data["lat"], location_data["lon"]],
            radius=1609.34 * 15,  # 15 miles in meters
            color="#007dc6",
            fill=True,
            fill_color="#007dc6",
            fill_opacity=0.1,
            popup="15 mile radius"
        ).add_to(m)
        
        return m

    def add_walmart_locations(self, m, competitor_locations):
        """Adds markers for existing Walmart locations."""
        for loc in competitor_locations:
            # Safe extraction of lat/lon
            lat = loc.get("lat") or loc.get("center", {}).get("lat")
            lon = loc.get("lon") or loc.get("center", {}).get("lon")
        
            if lat is None or lon is None:
                continue  # Skip if location is malformed

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.6,
                popup="Existing Walmart"
            ).add_to(m)

    def add_candidate_locations(self, map_obj: folium.Map, candidates: pd.DataFrame) -> None:
        """Add candidate locations to map"""
        for _, row in candidates.iterrows():
            folium.Marker(
                [row["lat"], row["lon"]],
                icon=folium.Icon(color="green", icon="star", prefix="fa"),
                tooltip=f"Score: {row['demand_score']:.2f}",
                popup=self._create_candidate_popup(row)
            ).add_to(map_obj)

    def _create_candidate_popup(self, candidate: Dict) -> str:
        """Create HTML popup for candidate locations"""
        return f"""
        <b>Potential Walmart Location</b><br>
        Address: {candidate.get('address', 'Unknown')}<br>
        Demand Score: {candidate.get('demand_score', 0):.2f}<br>
        Break-Even: {candidate.get('break_even_months', 0):.1f} months<br>
        Land Cost: ${candidate.get('price_per_sqft', 0):.2f}/sqft
        """

    def create_demand_chart(self, candidates: pd.DataFrame) -> px.bar:
        """Create demand score bar chart"""
        top_locations = candidates.nlargest(10, "demand_score")
        fig = px.bar(
            top_locations,
            x="address",
            y="demand_score",
            title="Top Locations by Demand Score",
            labels={"demand_score": "Demand Score", "address": "Location"},
            color="demand_score",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    def _calculate_zoom(self, locations: List[Tuple[float, float]]) -> int:
        """Calculate appropriate zoom level for given locations"""
        # Simplified calculation - in reality would use bounds
        return max(4, min(12, 14 - len(locations) // 10))