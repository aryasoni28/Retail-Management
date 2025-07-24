import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
from geopy.distance import geodesic
import requests
from constants import MODEL_DIR
from utils.data_fetcher import DataFetcher
from utils.geo_utils import GeoUtils
from utils.model_utils import ModelUtils

class AnalysisEngine:
    def __init__(self):
        """Initializes the engine, loading necessary utilities and the demand model."""
        self.data_fetcher = DataFetcher()
        self.geo = GeoUtils()
        self.model_utils = ModelUtils()
        self.demand_model = self._load_or_create_demand_model()

    def _load_or_create_demand_model(self):
        """Loads the demand prediction model, creating a placeholder if it doesn't exist."""
        model_dir = Path(r"C:\Users\aryar\Downloads\walmart_locator_rhn\models")
        model_path = model_dir / "demand_model.joblib"
        #model_path = Path(r"C:\Users\aryar\Downloads\walmart_locator_rhn\models")
    
        if not model_path.exists():
            print("⚠️ Model not found. Training a placeholder model with synthetic data.")
        
        # Generate synthetic data for a one-time training
            X = pd.DataFrame({
            "median_income": np.random.normal(65000, 20000, 500),
            "population": np.random.poisson(40000, 500),
            "competition_score": np.random.uniform(0.1, 1.0, 500),
        })
        
        # Synthetic target: demand is higher with more income/pop and less competition
            y = (X["median_income"] / 50000) + (X["population"] / 100000) + (X["competition_score"] * 0.5)
            y = np.clip(y / y.max(), 0.1, 0.99)  # Normalize to 0-1 range

            self.model_utils.train_demand_model(X, y, model_path)
    
        print("✅ Demand prediction model loaded.")
        return self.model_utils.load_model(model_path)
    def get_location_explanation(self, location_data: Dict) -> str:
        """
        Calls Gemini API to generate an explanation for why this location was selected.
        Returns a human-readable explanation string.
        """
        try:
            # Prepare the prompt with location data
            prompt = f"""
            Explain why this location would be a good choice for a new Walmart store based on these metrics:
            - Address: {location_data.get('address', 'Unknown')}
            - Population: {location_data.get('population', 0):,}
            - Median Income: ${location_data.get('median_income', 0):,}
            - Demand Score: {location_data.get('demand_score', 0):.2f}/1.0
            - Competition Score: {location_data.get('competition_score', 0):.2f}/1.0 (higher is better)
            - Break-even Time: {location_data.get('break_even_months', 0):.1f} months
            - Land Cost: ${location_data.get('price_per_sqft', 0):.2f}/sqft
            - Literacy Rate: {location_data.get('literacy_rate', 'N/A')}%
            - Major Occupation: {location_data.get('major_occupation', 'N/A')}
            
            Provide a concise 3-4 sentence explanation focusing on the most important factors.
            """
            
            # Call Gemini API (replace with your actual API key and endpoint)
            api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            api_key = "your_gemini_api_key_here"  # Should be stored securely
            
            response = requests.post(
                f"{api_url}?key={api_key}",
                json={
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Extract the generated text
            explanation = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return explanation or "Explanation not available"
            
        except Exception as e:
            print(f"Error getting explanation from Gemini API: {e}")
            return "Unable to generate explanation due to technical issues"
    def generate_candidate_locations(self, params: Dict) -> pd.DataFrame:
        """
        Generates candidate locations using a grid search and pre-filters by budget.
        """
        search_area = params['search_area_name']
        print(f"🌍 Starting grid search for {search_area}...")

        # Bounding boxes for grid search (can be expanded)
        search_bounds = {
            "Colorado": {"min_lat": 37.0, "max_lat": 41.0, "min_lon": -109.0, "max_lon": -102.0},
            "Texas": {"min_lat": 25.8, "max_lat": 36.5, "min_lon": -106.6, "max_lon": -93.5},
        }
        bounds = search_bounds.get(search_area)
        if not bounds:
            return pd.DataFrame()

        # Create a grid of lat/lon points
        grid_density = 20 # Higher density for more candidates
        lats = np.linspace(bounds["min_lat"], bounds["max_lat"], grid_density)
        lons = np.linspace(bounds["min_lon"], bounds["max_lon"], grid_density)
        
        budget_candidates = []
        max_land_cost = params["budget"]

        for lat in lats:
            for lon in lons:
                # Use estimated land value for quick budget filtering
                estimated_land_data = self.data_fetcher.get_land_values(lat, lon)
                price = estimated_land_data.get("price_per_sqft", float('inf'))

                if price <= max_land_cost:
                    budget_candidates.append({
                        "lat": lat, "lon": lon, "price_per_sqft": price
                    })
        
        print(f"💰 Found {len(budget_candidates)} candidates within budget.")
        if not budget_candidates:
            return pd.DataFrame()

        # Limit candidates to avoid long analysis times
        max_to_analyze = params.get("max_candidates", 100)
        if len(budget_candidates) > max_to_analyze:
            indices = np.random.choice(len(budget_candidates), max_to_analyze, replace=False)
            final_candidates = [budget_candidates[i] for i in indices]
        else:
            final_candidates = budget_candidates
            
        return pd.DataFrame(final_candidates)

    def analyze_locations(self, candidates: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Analyzes a dataframe of candidates, returning enriched data."""
        # Fetch competitor data specifically for the selected area
        search_area = params['search_area_name']
        competitor_locations = self.data_fetcher.get_walmart_locations(search_area)
        
        results = []
        total = len(candidates)
        for idx, row in candidates.iterrows():
            print(f"Analyzing location {idx + 1}/{total}...")
            # Pass the fetched competitors to the analysis function
            metrics = self._analyze_single_location(row.to_dict(), params, competitor_locations)
            if metrics:
                results.append(metrics)
        return pd.DataFrame(results)

    def _analyze_single_location(self, location: Dict, params: Dict, competitor_locations: List[Dict]) -> Optional[Dict]:
        """Performs a full analysis on a single candidate location."""
        try:
            # ... (keep the existing code for getting demographics)
            demo_data = self.data_fetcher.get_demographics(location["lat"], location["lon"])
            if not demo_data:
                print(f"Skipping location due to missing demographic data.")
                return None

            # 2. Calculate Competition using the pre-fetched data
            competition_score = self._calculate_competition(
                location["lat"], location["lon"], params["search_radius_miles"], competitor_locations
            )
            # 3. Predict Demand using the model
            model_features = {
                "median_income": demo_data["median_income"],
                "population": demo_data["population"],
                "competition_score": competition_score,
            }
            demand_score = self.model_utils.predict_with_fallback(
                self.demand_model, model_features, self._fallback_demand_prediction
            )

            # 4. Calculate Financials
            break_even = self._calculate_break_even(
                demand_score, location["price_per_sqft"], 40000 # Assume fixed store size (sqft)
            )

            # 5. Reverse geocode to get a readable address
            address = self.geo.get_address(location["lat"], location["lon"])
            break_even = self._calculate_break_even(demand_score, location["price_per_sqft"], 40000)
            return { #...
                **location,
                **demo_data,
                "address": self.geo.get_address(location["lat"], location["lon"]),
                "competition_score": competition_score,
                "demand_score": demand_score,
                "break_even_months": self._calculate_break_even(demand_score, location["price_per_sqft"], 40000),
            }
        except Exception as e:
            print(f"❗️ Analysis failed for location {location}: {e}")
            return None
    
    def _calculate_competition(self, lat: float, lon: float, radius_miles: float, competitor_locations: List[Dict]) -> float:
        """
        Calculates a competition score (0-1). Higher score means less competition.
        """
        if not competitor_locations:
            return 1.0 # No competitors found, max score

        nearby_competitors = 0
        for store in competitor_locations:
            # Handle different ways location data might be structured
            store_lat = store.get('lat') or store.get('center', {}).get('lat')
            store_lon = store.get('lon') or store.get('center', {}).get('lon')
            if not store_lat or not store_lon: continue

            dist = geodesic((lat, lon), (store_lat, store_lon)).miles
            if dist <= radius_miles:
                nearby_competitors += 1
        
        return 1.0 / (1.0 + nearby_competitors)
    def get_best_location_analysis(self, top_location: Dict) -> Dict:
        """Enhanced analysis for the #1 recommended location"""
        enhanced_data = self.data_fetcher.get_enhanced_demographics(
        top_location["lat"], 
        top_location["lon"]
    )
    
        if not enhanced_data:
        # Fallback to basic data
            return top_location
    
    # Calculate elderly percentage
        enhanced_data["elderly_pct"] = round(
        100 - enhanced_data["children_pct"] - enhanced_data["working_pct"], 
        1
    )
    
        return {**top_location, **enhanced_data}
    def _predict_demand(self, features: Dict) -> float:
        """Predicts demand score (0-1) using the loaded model."""
        input_df = pd.DataFrame([features])
        # Ensure column order matches training
        input_df = input_df[["median_income", "population", "competition_score"]]
        score = self.demand_model.predict(input_df)[0]
        return float(np.clip(score, 0.0, 1.0))
    
    def _fallback_demand_prediction(self, features: Dict) -> float:
        """Simple heuristic for demand if the ML model fails."""
        income_score = features.get('median_income', 50000) / 100000
        pop_score = features.get('population', 25000) / 100000
        comp_score = features.get('competition_score', 0.5)
        # Weighted average
        return np.clip(0.4 * income_score + 0.3 * pop_score + 0.3 * comp_score, 0, 1)

    def _calculate_break_even(self, demand_score: float, land_cost_sqft: float, store_size_sqft: float) -> float:
        """
        Calculates the estimated break-even time in months.
        This is a simplified financial model for demonstration.
        """
        # Costs
        land_investment = land_cost_sqft * store_size_sqft
        construction_cost = 20_000_000 # Fixed estimate for building a supercenter
        total_investment = land_investment + construction_cost
        
        monthly_op_cost = 500_000 # Staff, utilities, etc.
        
        # Revenue
        # Tie potential revenue to the demand score. A 1.0 score equals max potential.
        max_monthly_revenue = 3_000_000 
        estimated_monthly_revenue = max_monthly_revenue * demand_score
        
        monthly_profit = estimated_monthly_revenue - monthly_op_cost

        if monthly_profit <= 0:
            return np.inf  # Never breaks even

        return total_investment / monthly_profit
    # In class AnalysisEngine:

    def _generate_synthetic_demographics(self, location: Dict) -> Dict:
        """Generates plausible fake demographic data when the API fails."""
        # Create some variation based on location to make it look realistic
        np.random.seed(int(location['lat'] * 100)) # Make synthetic data consistent for the same point
        return {
            "population": np.random.randint(15000, 150000),
            "median_income": np.random.randint(45000, 95000)
        }

    def _analyze_single_location(self, location: Dict, params: Dict, competitor_locations: List[Dict]) -> Optional[Dict]:
        """Performs a full analysis on a single candidate location, with fallback."""
        try:
            # 1. Get Demographics
            demo_data = self.data_fetcher.get_demographics(location["lat"], location["lon"])
            
            # === START OF THE FORTIFICATION ===
            # If the API call fails, generate synthetic data instead of skipping.
            if not demo_data:
                print(f"⚠️ API failed for {location['lat']:.2f},{location['lon']:.2f}. Using synthetic demographic data.")
                demo_data = self._generate_synthetic_demographics(location)
            # === END OF THE FORTIFICATION ===

            # 2. Calculate Competition
            competition_score = self._calculate_competition(
                location["lat"], location["lon"], params["search_radius_miles"], competitor_locations
            )

            # 3. Predict Demand using the model
            model_features = {
                "median_income": demo_data["median_income"],
                "population": demo_data["population"],
                "competition_score": competition_score,
            }
            demand_score = self.model_utils.predict_with_fallback(
                self.demand_model, model_features, self._fallback_demand_prediction
            )

            # 4. Calculate Financials
            break_even = self._calculate_break_even(
                demand_score, location["price_per_sqft"], 40000 
            )

            # 5. Reverse geocode to get a readable address
            address = self.geo.get_address(location["lat"], location["lon"])
            
            return {
                **location,
                **demo_data,
                "address": address,
                "competition_score": competition_score,
                "demand_score": demand_score,
                "break_even_months": break_even,
            }
        except Exception as e:
            print(f"❗️ Analysis failed for location {location}: {e}")
            return None
