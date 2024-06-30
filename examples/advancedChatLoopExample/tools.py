from llmFunctionDecorator import tool
import json


@tool(
    enabled=True,                                                               # optional enabled argument
    purpose="Get the current weather in a given location.",                     # description of the function
    location=str,                                                               # type of the location argument
    location_description="The city and state, e.g. San Francisco, CA",          # description of the location argument
    unit=["celsius", "fahrenheit"],                                             # possible values for the unit argument
    unit_description="The unit of temperature, e.g. celsius or fahrenheit",     # description of the unit argument
    required=["location"],                                                      # required arguments
)
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


@tool(
    enabled=True,                                                               # optional enabled argument
    purpose="Get the current time in a given location.",                        # description of the function
    location=str,                                                               # type of the location argument
    location_description="The city and state, e.g. San Francisco, CA",          # description of the location argument
    required=["location"],                                                      # required arguments
)
def get_current_time(location):
    """Get the current time in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": location, "time": "3:00 PM"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": location, "time": "12:00 PM"})
    elif "paris" in location.lower():
        return json.dumps({"location": location, "time": "9:00 PM"})
    else:
        return "I don't know the time in " + location
