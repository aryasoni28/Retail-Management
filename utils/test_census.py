import requests
from config_manager import ConfigManager

def test_census_api():
    config = ConfigManager()
    api_key = config.get_api_key("census")
    
    # Test with known Colorado county (Denver)
    url = "https://api.census.gov/data/2021/acs/acs5"
    params = {
        "get": "B01003_001E,B19013_001E",
        "for": "county:031",
        "in": "state:08",
        "key": c7628f551572c7f4af83313467b7e11fa569a185
    }
    
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(response.json())

if __name__ == "__main__":
    test_census_api()