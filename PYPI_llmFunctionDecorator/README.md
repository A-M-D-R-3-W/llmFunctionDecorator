# llmFunctionDecorator

A Python package designed to simplify the process of creating and managing function calls to OpenAI's API, as well as models using LiteLLM's API framework.

## Examples

Examples (A separate readme and project files) are located in the `/examples` folder of the Github repo.

## Installation

`llmFunctionDecorator` is available on PyPi, so installation is easy.

```python
pip install llmFunctionDecorator
```

Upon installation, make sure to import the package into your project.

```python
from llmFunctionDecorator import tool, FunctionRegistry
```

Note: Not required, but I also recommend using `print` from [Rich](https://github.com/Textualize/rich) for nice JSON formatting among other things. Rich is used in the attached example.

```python
from rich import print
```

## Quickstart

### 1. Defining Functions

First, define your functions following standard Python syntax, in the same way OpenAI and LiteLLM expect. For example:

```python
def an_awesome_function(variable1, variable2="A default value for variable2"):
    # Function body ...
    return desiredVariable
```

### 2. Decorating your Functions

Use the `@tool` decorator to create your function descriptions which will be passed to the LLM. Add relevant details such as purpose, parameters, and descriptions. Make sure the decorator is ***directly above*** your function as shown.

The description will look something like this:
```python
@tool(
    enabled=True,
    purpose="An awesome function that does something amazing.",
    variable1=int,
    variable1_description="The first variable that will be used to do some awesome thing.",
    variable2=["option1", "option2"],
    variable2_description="The second variable that will be used to do another awesome thing.",
    required=["variable1", "variable2"]
)
def an_awesome_function(variable1, variable2="A default value for variable2"):
    # Function body ...
    return desiredVariable
```

### 3. Submitting Your Functions to the API

Using the functions is very simple. In your request, under the `tools` key, set it to `FunctionRegistry.tools()`.

Also, you must set the `tool_choice` key to `FunctionRegistry.tool_choice()`. More information on `FunctionRegistry.tool_choice()` is provided in the ***FunctionRegistry Class*** section of this readme.

```python
response = litellm.completion(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=FunctionRegistry.tools(),
            tool_choice=FunctionRegistry.tool_choice(),
)
```

Note: If you want to avoid using `FunctionRegistry.tool_choice()` and would rather manually input values for `tool_choice`, you will need to format your response as below, otherwise it will cause errors.

```python
if FunctionRegistry.tools():
    response = litellm.completion(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=FunctionRegistry.tools(),
        tool_choice="auto",
    )
else:
    response = litellm.completion(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )
)
```

This will maintain your access to `tool_choice` if there are functions enabled, and if there are no functions, it will use the second response format.

## Detailed Look at Creating Your `@tool` Decorator Function Descriptions

### Required and Optional Parameters:

- `purpose` (Required, `str`): A brief description of what the wrapped function does. This should be a human-readable string that clearly communicates the function's purpose.
  ```python
  purpose="Get the current weather in a given location"
  ```
  
- `enabled` (Optional, `bool`): Flag indicating if the function is enabled. If not present, will default to `True`.
  ```python
  enabled=False
  ```
  
- `required` (Optional, `list` of `str`): A list of parameter names that are required for the function to operate. This is useful for specifying which parameters cannot be omitted when calling the function.
  ```python
  required=["location", "unit"]
  ```

### Parameter Keyword Arguments (Dynamic):
- `**kwargs`: In addition to the parameters mentioned above, you can specify any number of additional keyword arguments. These are used to define the parameters (variables) that the decorated function takes as input. The keys should be the names of the parameters, and the values should define their types or allowable values (for enums).
  
  For each parameter of the function assigned to `function_ref`, you can provide:
  - The parameter type by simply specifying a Python type (e.g., `str`, `int`, etc.) as the value that the parameter expects as input.
    ```python
    location=str
    ```
  - For enum parameters, instead of a single type, you provide a list of allowable values (e.g., `["celsius", "fahrenheit"]` for a temperature unit parameter).
    ```python
    unit=["celsius", "fahrenheit"]
    ```
  - You can also append `_description` to any parameter name (e.g., `location_description`) as an additional key to provide a human-readable description of what that parameter is for. ***Every parameter should have an accompanying description key.***
    ```python
    location_description="The city and state, e.g. San Francisco, CA"
    ```

    Putting this all together, we get our function description.
```python
@tool(
    enabled=True,                                                               # optional enabled argument
    purpose="Get the current weather in a given location.",                     # description of the function
    location=str,                                                               # type of the location argument
    location_description="The city and state, e.g. San Francisco, CA",          # description of the location argument
    unit=["celsius", "fahrenheit"],                                             # possible values for the unit argument
    unit_description="The unit of temperature, e.g. celsius or fahrenheit",     # description of the unit argument
    required=["location"],                                                      # required arguments
)
```

### Permissible Data Types
The following is a list of data types that can be assigned to a parameter (variable).
- variable1=`int`
- variable1=`float`
- variable1=`str`
- variable1=`bool`
- variable1=`list`
- variable1=`tuple`
- variable1=`dict`
- variable1=`None`
  
In addition, enums can be assigned int, float, str, or bool.
For example,

- variable1=`[12, 19, 17]`
- variable1=`[18.6, 78.2, 97.0]`
- variable1=`["first", "second", "last"]`
- variable1=`[True, False]`

Enums can also have various data types. For example,
- variable1=`[15, 17.2, "hello", True]`


## FunctionRegistry Class

The `FunctionRegistry` class acts as a storage and management system for all registered functions, enabling the dynamic invocation of these functions with arguments specified at runtime.

### Overview

The `FunctionRegistry` provides several class methods for managing functions, their metadata, and invocation:

- `register_function(name, tool_instance)`: Automatically called by the `ToolWrapper` when a function is decorated with `@tool`. It registers a function with its metadata in the registry. This will never* need to be called directly as it is automatically handled.
  
- `get_registry()`: Returns a dictionary of all enabled function instances that are registered. Useful for inspecting which functions are available for invocation.
  
- `tools()`: Retrieves registered `ToolWrapper` instances, converts them to their dictionary representations, and returns a list of these dictionaries. This is the method you will pass to the `tools` key when creating your response, i.e., this will replace the JSON in your API call.
  
- `registry_status()`: Returns a string summary of all registered functions along with their enabled status. This is helpful for debugging purposes to ensure that the intended functions are enabled and correctly registered.
  
- `call_function(name, **kwargs)`: Tries to invoke a registered function by its name, passing the provided keyword arguments. I.e., it dynamically invokes functions based on LLM requests. See the **Parallel Function Call** example to see it in action, and understand how to implement it.

### Usage

Although most interaction with `FunctionRegistry` is automated through the use of the `@tool` decorator, a few of these methods can be very useful, especially for debugging or extending the capabilities of your LLM integration. Here are some examples:

**Checking Registered Functions:**

```python
from llmFunctionDecorator import FunctionRegistry

# List all registered functions and their enabled statuses
print(FunctionRegistry.registry_status())
```

**Directly Invoking a Registered Function:**

This example demonstrates how you might directly invoke a function that has been registered with the FunctionRegistry. 

```python
result = FunctionRegistry.call_function('get_current_weather', location="Tokyo, Japan")
print(result)
```

This would attempt to call the `get_current_weather` function (assuming it's registered and enabled) with the specified location parameter. As you'll see in the **Parallel Function Call** example, this is handled by the LLM and will not require manual entry.

**Listing Functions for LLM Integration:**

When preparing a request to an LLM, you will need to include the list of functions. Here's how you can retrieve this list in the format expected by the LLM:

```python
from llmFunctionDecorator import FunctionRegistry

tools_list = FunctionRegistry.tools()
if tools_list:
    response = litellm.completion(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools_list,
        tool_choice=FunctionRegistry.tool_choice(),
    )
```

This snippet retrieves the list of registered and enabled function tools, passing it along to the LLM as part of the completion request.

The `FunctionRegistry.tool_choice()` has 3 possible states:
1. If there are no functions in the registry (or all functions are disabled), `tool_choice()` will return `None`, telling the LLM that there are no functions to call and it should continue the conversation.
2. If there is at least 1 function in the registry (and it's enabled), `tool_choice()` will return `"auto"`, telling the LLM to decide if a function call is needed depending on the user's message.
3. If you want to force a function call to a particular function, you can pass the function as an input.
   I.e., `FunctionRegistry.tool_choice(get_current_weather)` will return `{'type': 'function', 'function': {'name': 'get_current_weather'}}`, forcing the LLM to call that function.
   Note: Make sure you are passing the actual function, not contained in a string. This only accepts a single function that is present in the registry and enabled.

### Best Practices

- Regularly use `registry_status()` during development to verify that your functions are correctly registered and in the state (enabled/disabled) you expect.
- Utilize direct invocation with `call_function()` for testing your functions within the Python environment before integrating with LLM.
- Keep your registered functions' interfaces simple and consistent to ensure smooth dynamic invocation by the LLM.
