from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    if city.lower() == "london":
        return "It's cloudy and 15°C in London."
    elif city.lower() == "islamabad":
        return "It's sunny and 35°C in Islamabad."
    else:
        return "Weather data not available for this city."