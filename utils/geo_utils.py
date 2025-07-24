from typing import Tuple, Optional
from functools import lru_cache
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pycountry
import requests

class GeoUtils:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="walmart_site_selector_ai")
        # Add a rate limiter to avoid hitting API limits
        self.reverse = RateLimiter(self.geolocator.reverse, min_delay_seconds=1, error_wait_seconds=5)

    @lru_cache(maxsize=512)
    def get_location_details(self, lat: float, lon: float) -> Optional[dict]:
        """Cachable reverse geocoding to get all location details."""
        try:
            return self.reverse((lat, lon), language='en', addressdetails=True, timeout=10)
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            print(f"Geocoding error for ({lat}, {lon}): {e}")
            return None

    def get_address(self, lat: float, lon: float) -> str:
        location = self.get_location_details(lat, lon)
        return location.address if location else "Address not found"

    def get_state_code(self, lat: float, lon: float) -> Optional[str]:
        """Gets the 2-letter state code (e.g., 'CO')."""
        location = self.get_location_details(lat, lon)
        if location:
            address = location.raw.get('address', {})
            state_name = address.get('state')
            if state_name:
                try:
                    # pycountry can find subdivision by name
                    subdivision = pycountry.subdivisions.lookup(state_name)
                    return subdivision.code.split('-')[1]
                except LookupError:
                    return None
        return None

    def get_fips_codes(self, lat: float, lon: float) -> Optional[Tuple[str, str]]:
        """
        Gets state and county FIPS codes from FCC API.
            Returns a tuple (state_fips, county_fips) or None if lookup fails.  """
        try:
            url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

        # FCC API doesn't use 'status'; just check if expected keys exist
            if "State" in data and "County" in data:
                state_fips = data["State"]["FIPS"]
                county_fips = data["County"]["FIPS"][-3:]  # last 3 digits
                return state_fips, county_fips
            else:
                print(f"FCC API returned incomplete data for ({lat}, {lon}): {data}")
        except Exception as e:
            print(f"FIPS lookup failed for ({lat}, {lon}): {e}")
        return None
