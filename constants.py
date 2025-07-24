from enum import Enum
import os
from pathlib import Path

class APISources(Enum):
    OPENROUTE = "openroute"
    US_CENSUS = "census"
    OSM = "openstreetmap"
    ZILLOW = "zillow"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config.yaml"

# Defaults for the UI and analysis
DEFAULT_CONFIG = {
    "analysis": {
        "budget": 80,             # $/sqft
        "break_even_months": 36,  # 3 years
        "target_turnover": 20,    # In millions
        "search_radius_miles": 15,
        "max_candidates": 100
    },
    "visualization": {
        "map_center": [39.8283, -98.5795],  # US Center
        "default_zoom": 4
    }
}

# US Census Variables to fetch
# B01003_001E: Total Population
# B19013_001E: Median Household Income
# US Census Variables to fetch (2021 ACS 5-year)
CENSUS_VARIABLES = {
    "population": "B01003_001E",  # Total population
    "median_income": "B19013_001E",  # Median household income
    "literacy_rate": "B15003_017E,B15003_018E,B15003_019E,B15003_020E,B15003_021E,B15003_022E,B15003_023E,B15003_024E,B15003_025E",  # 9th grade to PhD
    "high_school_complete": "B15003_017E",  # Specifically 10th grade pass
    "children_pop": "B01001_003E,B01001_004E,B01001_027E,B01001_028E",  # Male+Female 0-14
    "working_pop": "B01001_005E,B01001_006E,B01001_007E,B01001_008E,B01001_009E,B01001_010E,B01001_011E,B01001_012E,B01001_013E,B01001_029E,B01001_030E,B01001_031E,B01001_032E,B01001_033E,B01001_034E,B01001_035E,B01001_036E",  # Male+Female 15-64
    "top_occupation": "C24050_002E,C24050_003E,C24050_004E,C24050_005E"
}

# Mock state land price ranges ($ per sqft) for demonstration
# In a real scenario, this would come from a dedicated API.
STATE_LAND_PRICES = {
    "CA": (150, 400),
    "NY": (120, 350),
    "TX": (40, 150),
    "FL": (60, 200),
    "IL": (50, 180),
    "CO": (70, 250), # Added Colorado
    "default": (30, 120)
}