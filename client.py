import requests

URL = "http://127.0.0.1:8000/data"

def fetch_data():
    """Fetches network data from API."""
    response = requests.get(URL)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data.")
        return None

if __name__ == "__main__":
    data = fetch_data()
    print(data)
