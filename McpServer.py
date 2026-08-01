"""
mcp_server.py — MCP server exposing weather and soil tools for AI Crop Doctor.
"""
import asyncio
import os
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
weather_api_key = os.getenv("OPENWEATHER_API_KEY")

mcp = FastMCP("crop-doctor-tools")

# Same mock lookup table that used to live in app.py's get_soil_condition().
SOIL_DATABASE = {
    "lucknow": "Alluvial soil, rich in potash but poor in phosphorus",
    "punjab": "Loamy to sandy loam, good for wheat and rice",
    "maharashtra": "Black cotton soil, retains moisture well",
    "kerala": "Laterite soil, acidic in nature",
    "gujarat": "Sandy loam to clay loam, moderate fertility",
    "rajasthan": "Sandy and arid soil, low organic matter",
    "karnataka": "Red laterite soil, slightly acidic",
    "andhra pradesh": "Black and red loamy soil",
    "tamil nadu": "Red loam and alluvial, varies by region",
    "west bengal": "Alluvial soil, good for paddy cultivation",
}


@mcp.tool()
def get_weather(location: str) -> str:
    """Fetches current weather condition for a given location.
    Args:
        location: City name
    Returns:
        A short human-readable weather summary.
        Falls back to mock data if no OpenWeatherMap API key is configured, or if the request fails.
    """
    if not weather_api_key:
        return "Temperature: 32°C, Humidity: 65%"

    try:
        url = (
            "http://api.openweathermap.org/data/2.5/weather"
            f"?q={location}&appid={weather_api_key}&units=metric"
        )
        res = requests.get(url, timeout=10).json()
        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        desc = res["weather"][0]["description"]
        return f"Temperature: {temp}°C, Humidity: {humidity}%, Condition: {desc.capitalize()}"
    except Exception:
        return "Weather data unavailable."


@mcp.tool()
def get_soil_condition(location: str) -> str:
    """Look up the soil typefor a given Indian state or city.
    Args:
        location: City or state name
    Returns:
        A short description of the typical local soil type.
        Falls back to a generic default if the location isn't in the lookup table.
    """
    city = location.lower().split(",")[0].strip()
    return SOIL_DATABASE.get(city, "Standard Loamy Soil")


if __name__ == "__main__":
    mcp.run(transport="stdio")