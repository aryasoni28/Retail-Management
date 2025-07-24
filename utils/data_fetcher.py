import requests
import numpy as np
from typing import Optional, Dict, List, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from constants import APISources, CENSUS_VARIABLES, STATE_LAND_PRICES
from utils.config_manager import ConfigManager
from utils.geo_utils import GeoUtils

class DataFetcher:
    def __init__(self):
        self.config = ConfigManager()
        self.session = self._create_session()
        self.geo = GeoUtils()
        self.census_api_key = self.config.get_api_key("census")

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "WalmartSiteSelector/1.0"})
        return session

    def _make_api_request(self, url: str, params: dict = None) -> Optional[dict]:
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
        
        # Debug logging
            print(f"API Response Status: {response.status_code}")
            print(f"Response Sample: {response.text[:200]}...")
        
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error ({url}): {str(e)}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text[:200]}")
            return None

    def get_demographics(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetches demographic data from the US Census API with proper error handling"""
        if not self.census_api_key:
            print("⚠️ Census API key not found. Using synthetic data.")
            return None

    # Get FIPS codes first
        fips = self.geo.get_fips_codes(lat, lon)
        if not fips:
            print(f"⚠️ Couldn't get FIPS codes for {lat},{lon}. Using synthetic data.")
            return None

        state_fips, county_fips = fips
    
    # Use the 2021 ACS 5-year estimates (more stable than 2022)
        base_url = "https://api.census.gov/data/2021/acs/acs5"
    
        params = {
        "get": ",".join(CENSUS_VARIABLES.values()),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
        "key": self.census_api_key
    }
    
        try:
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
        
        # Additional validation
            if not response.text.strip():
                print("Empty API response")
                return None
            
            data = response.json()
        
            if not data or len(data) < 2:
                print("Incomplete API response")
                return None
            
        # Extract data
            header = data[0]
            values = data[1]
            return {
                "population": int(values[header.index("B01003_001E")]),
            "median_income": int(values[header.index("B19013_001E")])
        }
        
        except Exception as e:
            print(f"⚠️ Census API failed: {str(e)}")
            return None
    def get_enhanced_demographics(self, lat: float, lon: float) -> Dict:
        """Fetch detailed demographics for the best location only"""
        if not self.census_api_key:
            return None

        fips = self.geo.get_fips_codes(lat, lon)
        if not fips:
            return None

        state_fips, county_fips = fips
    
    # Base population query
        base_params = {
        "get": "B01003_001E,B15003_022E",
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
        "key": self.census_api_key
    }
        edu_params = {
        "get": "B15003_001E,B15003_017E,B15003_018E,B15003_019E,B15003_020E,B15003_021E,B15003_022E,B15003_023E,B15003_024E,B15003_025E",
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
        "key": self.census_api_key
    }
    
        edu_data = self._make_api_request(
        "https://api.census.gov/data/2021/acs/acs5",
        edu_params
    )
    
    # Occupation-specific query
        occupation_params = {
        "get": "C24050_002E,C24050_003E,C24050_004E,C24050_005E",
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
        "key": self.census_api_key
    }

        try:
        # Get base demographics
            base_data = self._make_api_request(
                "https://api.census.gov/data/2021/acs/acs5",
                base_params
            )
            if edu_data and len(edu_data) > 1:
                total_pop_25plus = int(edu_data[1][0])
        # Sum everyone who completed 10th grade or higher (variables 2-10 in response)
                educated_pop = sum(int(x) for x in edu_data[1][1:10])
                literacy_rate = round((educated_pop / total_pop_25plus) * 100, 1)
            else:
                literacy_rate = None
            
        # Get occupation data
            occupation_data = self._make_api_request(
            "https://api.census.gov/data/2021/acs/acs5",
            occupation_params
        )

            if not base_data or not occupation_data:
                return None

        # Process base data
            total_pop = int(base_data[1][0])
            literate_pop = int(base_data[1][1])
        
        # Process occupation data (get the largest sector)
            occupations = {
            "Management": int(occupation_data[1][0]),
            "Service": int(occupation_data[1][1]),
            "Sales": int(occupation_data[1][2]),
            "Construction": int(occupation_data[1][3])
        }
            major_occupation = max(occupations, key=occupations.get)

            return {
            "literacy_rate": literacy_rate,
            "major_occupation": major_occupation,
            "children_pct": self._get_age_group_percentage(lat, lon, "0-15"),
            "working_pct": self._get_age_group_percentage(lat, lon, "15-65"),
            # Elderly calculated by subtraction
        }
        
        except Exception as e:
            print(f"Enhanced demographics failed: {e}")
            return None

    def _get_age_group_percentage(self, lat: float, lon: float, age_group: str) -> float:
        """Helper method to get age group percentages"""
    # Implement actual API call here using similar pattern
    # Mock implementation for demonstration:
        if age_group == "0-15":
            return 18.5  # Example value
        elif age_group == "15-65":
            return 65.3  # Example value
        return 0.0
    def get_land_values(self, lat: float, lon: float) -> Dict:
        """
        Mocks fetching land values. In a real app, this would call a service like Zillow's API.
        """
        state_code = self.geo.get_state_code(lat, lon)
        price_range = STATE_LAND_PRICES.get(state_code, STATE_LAND_PRICES["default"])
        
        return {
            "price_per_sqft": round(np.random.uniform(*price_range), 2),
            "source": "State-level estimate"
        }

   # In class DataFetcher:
    # In data_fetcher.py
    def get_location_explanation(self, location_data: Dict) -> str:
        """Calls Gemini API to generate an explanation for the location recommendation"""
        try:
            api_key = self.config.get_api_key("gemini")
            if not api_key:
                return "Explanation unavailable (missing API key)"
            
            prompt = f"""
            Analyze why this location would be ideal for a new Walmart store based on:
            - Population: {location_data.get('population', 0):,}
            - Median Income: ${location_data.get('median_income', 0):,}
            - Demand Score: {location_data.get('demand_score', 0):.2f}/1.0
            - Competition: {location_data.get('competition_score', 0):.2f}/1.0
            - Break-even: {location_data.get('break_even_months', 0):.1f} months
            - Land Cost: ${location_data.get('price_per_sqft', 0):.2f}/sqft
        
            Provide a concise 3-4 sentence explanation focusing on the key strengths.
            """
        
            response = self.session.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            params={"key": api_key},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            timeout=10
        )
            response.raise_for_status()
        
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No explanation generated")
        
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Explanation unavailable due to technical issues"
    def get_walmart_locations(self, search_area_name: str) -> List[Dict]:
        """Fetches existing Walmart locations for a specific state to avoid timeouts."""
        print(f"🛒 Fetching existing Walmart locations for {search_area_name}...")
        
        # This query is now targeted to a specific state (admin_level=4 in OSM for US states)
        query = f"""
        [out:json][timeout:30];
        area[name="{search_area_name}"]->.searchArea;
        (
          node["brand"="Walmart"]["brand:wikidata"="Q483551"](area.searchArea);
          way["brand"="Walmart"]["brand:wikidata"="Q483551"](area.searchArea);
          relation["brand"="Walmart"]["brand:wikidata"="Q483551"](area.searchArea);
        );
        out center;
        """
        data = self._make_api_request("https://overpass-api.de/api/interpreter", {"data": query})
    
        if not data or not data.get('elements'):
            print(f"⚠️ Overpass API failed or returned no data for {search_area_name}. Using fallback locations.")
            return [
                {"lat": 39.7392, "lon": -104.9903, "tags": {"name": "Walmart Supercenter (Fallback)"}},
            ]
    
        locations = data.get('elements', [])
        print(f"✅ Found {len(locations)} competitor locations in {search_area_name}.")
        return locations