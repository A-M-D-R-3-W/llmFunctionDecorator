'''Modified example from https://platform.openai.com/docs/guides/function-calling for use with llmFunctionWrapper'''

from openai import OpenAI
import json
from llmFunctionWrapper import ToolWrapper, FunctionRegistry
from rich import print

client = OpenAI()

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

weatherFunction = ToolWrapper(
    function_ref=get_current_weather,                                           # function reference
    purpose="Get the current weather in a given location.",                     # description of the function
    location=str,                                                               # type of the location argument
    location_description="The city and state, e.g. San Francisco, CA",          # description of the location argument
    unit=["celsius", "fahrenheit"],                                             # possible values for the unit argument
    unit_description="The unit of temperature, e.g. celsius or fahrenheit",     # description of the unit argument
    required=["location"],                                                      # required arguments
)


def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]

    # Add the function description to the tools list and serialize
    unserializedTools = [weatherFunction]
    tools = [tool.to_dict() for tool in unserializedTools]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            # Use the FunctionRegistry to call the function and pass the arguments
            function_response = FunctionRegistry.call_function(
                function_name,
                **function_args
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response
print(run_conversation())
