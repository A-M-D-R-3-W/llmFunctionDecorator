'''Modified example from https://litellm.vercel.app/docs/completion/function_call for use with llmFunctionWrapper'''

import litellm
import json
import os
from llmFunctionWrapper import ToolWrapper, FunctionRegistry
from rich import print

# set openai api key
os.environ['OPENAI_API_KEY'] = "" # litellm reads OPENAI_API_KEY from .env and sends the request


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

weatherFunction = ToolWrapper(
    function_ref=get_current_weather,                                           # function reference
    purpose="Get the current weather in a given location.",                     # description of the function
    location=str,                                                               # type of the location argument
    location_description="The city and state, e.g. San Francisco, CA",          # description of the location argument
    unit=["celsius", "fahrenheit"],                                             # possible values for the unit argument
    unit_description="The unit of temperature, e.g. celsius or fahrenheit",     # description of the unit argument
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

timeFunction = ToolWrapper(
    function_ref= get_current_time,                                             # function reference
    purpose="Get the current time in a given location.",                        # description of the function
    location=str,                                                               # type of the location argument
    location_description="The city and state, e.g. San Francisco, CA",          # description of the location argument
    required=["location"],                                                      # required arguments
)


def test_parallel_function_call():

    # This will print the list of functions in the registry
    print("\nFunctions present in FunctionRegistry:", FunctionRegistry.get_registry())

    try:
        # Step 1: send the conversation and available functions to the model
        messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris? Also, please tell me the time in San Francisco as well."}]

        # Add the function descriptions to the tools list and serialize
        unserializedTools = [weatherFunction, timeFunction]
        tools = [tool.to_dict() for tool in unserializedTools]

        response = litellm.completion(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        print("\nFirst LLM Response:\n", response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print("\n\nLength of tool calls", len(tool_calls))

        # Step 2: check if the model wanted to call a function
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

                # This will print each response of the function calls
                print("\nFunction response:\n", function_response)

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = litellm.completion(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )  # get a new response from the model where it can see the function response
            print("\n\nSecond LLM response:\n", second_response)
            return second_response
    except Exception as e:
      print(f"Error occurred: {e}")

test_parallel_function_call()
