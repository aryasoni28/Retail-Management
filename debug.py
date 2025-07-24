import time
import pandas as pd
from analysis_engine import AnalysisEngine
from typing import Tuple
from utils.visualization import MapVisualizer

def debug_test():
    print("\n🔍 Starting DEBUG TEST V3")
    start_time = time.time()
    
    params = {
        "search_area_name": "Colorado",
        "budget": 150,
        "break_even_months": 24,
        "turnover": 1000000,
        "search_radius_miles": 50,
        "max_candidates": 5  # Test with 5 locations
    }

    print("\n1. Testing Engine Initialization...")
    engine = AnalysisEngine()
    print(f"✅ Engine loaded in {time.time()-start_time:.2f}s")
    
    print("\n2. Testing Location Generation...")
    candidates = engine.generate_candidate_locations(params)
    print(f"🌍 Generated {len(candidates)} locations in {time.time()-start_time:.2f}s")
    print("Sample location:", candidates.iloc[0].to_dict() if not candidates.empty else "❌ Empty")
    
    print("\n3. Testing Analysis...")
    results = engine.analyze_locations(candidates, params)
    print(f"🎯 Analyzed {len(results)} locations in {time.time()-start_time:.2f}s")
    
    if not results.empty:
        print("\n4. Testing Visualization...")
        visualizer = MapVisualizer()
        m = visualizer.create_base_map(results[["lat", "lon"]].values.tolist())
        
        # Get competitor locations
        competitors = engine.data_fetcher.get_walmart_locations(params['search_area_name'])
        visualizer.add_walmart_locations(m, competitors)
        visualizer.add_candidate_locations(m, results.head(3))  # Test with top 3
        
        # Save map to HTML for inspection
        m.save("debug_map.html")
        print("✅ Map visualization saved to debug_map.html")
        
        print("\nTop Results:")
        print(results[["address", "demand_score", "break_even_months"]].head())
    else:
        print("\n❌ No valid results - check logs above")

if __name__ == "__main__":
    debug_test()