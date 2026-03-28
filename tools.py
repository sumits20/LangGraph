from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return weather information for a city."""
    weather = {
        "london": "Cloudy, 14°C",
        "mumbai": "Sunny, 32°C",
        "delhi": "Hot, 36°C",
    }
    return weather.get(city.lower(), f"No weather data found for {city}.")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

@tool
def get_fun_fact(topic: str) -> str:
    """Return a short dummy fun fact about a topic."""
    facts = {
        "space": "Space is completely silent because sound needs a medium to travel.",
        "ocean": "More than 80% of the ocean is still unexplored.",
        "python": "Python was named after Monty Python, not the snake.",
    }
    return facts.get(topic.lower(), f"No fun fact found for {topic}.")
