import requests
from utils.config_manager import ConfigManager

def test_census_api():
    config = ConfigManager()
    api_key = config.get_api_key("census")  # Make sure your .env or config file contains the correct key
    
    url = "https://api.census.gov/data/2021/acs/acs5"
    params = {
        "get": "B01003_001E,B19013_001E",  # Total Population and Median Household Income
        "for": "county:031",               # Denver County
        "in": "state:08",                  # Colorado (state code 08)
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(response.json())

if __name__ == "__main__":
    test_census_api()
