# debug_gemini.py
import os
from utils.config_manager import ConfigManager

def debug_api_key():
    print("🔍 Debugging Gemini API Key Configuration")
    
    # Test direct environment variable
    env_key = os.getenv("GEMINI_KEY")
    print(f"Environment Variable (GEMINI_KEY): {'*****' if env_key else 'Not found'}")
    
    # Test config file
    config = ConfigManager()
    config_key = config.get_api_key("gemini")
    print(f"Config File Key: {'*****' if config_key else 'Not found'}")
    
    # Test actual API access
    if config_key:
        test_api_connection(config_key)

def test_api_connection(api_key: str):
    import requests
    print("\nTesting API connection...")
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            params={"key": api_key},
            json={
                "contents": [{
                    "parts": [{
                        "text": "Hello, respond with just 'OK' if you receive this."
                    }]
                }]
            },
            timeout=10
        )
        if response.status_code == 200:
            print("✅ API Connection Successful!")
            print("Response:", response.json())
        elif response.status_code == 400:
            print("❌ Bad Request - Check your API key format")
            print("Response:", response.text)
        elif response.status_code == 403:
            print("❌ Access Denied - Check if API is enabled in Google Cloud Console")
            print("Response:", response.text)
        else:
            print(f"❌ Unexpected Status Code: {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"❌ Connection Failed: {str(e)}")

if __name__ == "__main__":
    debug_api_key()