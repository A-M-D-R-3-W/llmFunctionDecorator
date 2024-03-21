# llmFunctionWrapper

A Python package designed to simplify the process of creating and managing function calls to OpenAI's API, as well as models using LiteLLM's API framework.

![Demo_llmFunctionWrapper](https://github.com/A-M-D-R-3-W/llmFunctionWrapper/assets/84816543/a7c0f6d8-9bbc-4ea4-a09d-9c709beed7fd)

## Examples

Examples (A separate readme and project files) are located in the `examples/` folder [here](examples/)

## Installation

`llmFunctionWrapper` is available on PyPi, so installation is easy.

```python
pip install llmFunctionWrapper
```

Upon installation, make sure to import the package into your project.

```python
from llmFunctionWrapper import ToolWrapper, FunctionRegistry
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

### 2. Wrapping your Functions

Use the `ToolWrapper` class to create your function descriptions which will be passed to the LLM. Add relevant details such as purpose, parameters, and descriptions.

The description will look something like this:
```python
awesomeFunction = ToolWrapper(
    function_ref=an_awesome_function,
    purpose="An awesome function that does something amazing.",
    variable1=int,
    variable1_description="The first variable that will be used to do some awesome thing.",
    variable2=["option1", "option2"],
    variable2_description="The second variable that will be used to do another awesome thing.",
    required=["variable1", "variable2"]
)
```

### 3. Submitting Your Functions to the API

Before you make your API request, you must serialize your function descriptions in OpenAI and LiteLLM's tool format.
```python
unserializedTools = [awesomeFunction] # If you have multiple functions, their descriptions must all be listed here (If you want to use them). Ex. unserializedTools = [awesomeFunction, otherFunction]
tools = [tool.to_dict() for tool in unserializedTools]
```
Alternatively, you can serialize each function individually in-line:
```python
tools = [awesomeFunction.to_dict(), otherFunction.to_dict()]
```
After serializing, submit your `tools` list in your request, in the same way as before.
```python
response = litellm.completion(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=tools,
            tool_choice="auto",
)
```

## Detailed Look at Creating Your ToolWrapper() Function Descriptions

### Required and Optional Parameters:
- `function_ref` (Required, `callable`): The actual Python function that this wrapper is meant to represent, and which will be called by the LLM. This parameter must be a callable object (e.g., a function or a method).
  ```python
  function_ref=get_current_weather
  ```

- `purpose` (Required, `str`): A brief description of what the wrapped function does. This should be a human-readable string that clearly communicates the function's purpose.
  ```python
  purpose="Get the current weather in a given location"
  ```
  
- `required` (Optional, `list` of `str`): A list of parameter names that are required for the function to operate. This is useful for specifying which parameters cannot be omitted when calling the function.
  ```python
  required=["location", "unit"]
  ```

### Parameter Keyword Arguments (Dynamic):
- `**kwargs`: In addition to the parameters mentioned above, you can specify any number of additional keyword arguments. These are used to define the parameters (variables) that the function assigned to `function_ref` takes. The keys should be the names of the parameters, and the values should define their types or allowable values (for enums).
  
  For each parameter of the function assigned to `function_ref`, you can provide:
  - The parameter type by simply specifying a Python type (e.g., `str`, `int`, etc.) as the value that the parameter expects as input.
    ```python
    location=str
    ```
  - For enum parameters, instead of a single type, you provide a list of allowable values (e.g., `["celsius", "fahrenheit"]` for a temperature unit parameter).
    ```python
    unit=["celsius", "fahrenheit"]
    ```
  - You can also append `_description` to any parameter name (e.g., `location_description`) as an additional key to provide a human-readable description of what that parameter is for. ***⚠️Every parameter should have an accompanying description key.⚠️***
    ```python
    location_description="The city and state, e.g. San Francisco, CA"
    ```

    Putting this all together, we get our function description.
```python
weatherFunction = ToolWrapper(
    function_ref=get_current_weather,                                           # function reference
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

## FunctionRegistry

The `FunctionRegistry` class serves as a central repository for all functions that can be called by the LLM. It ensures that functions are uniquely identified by their names and can be invoked dynamically with arguments specified at runtime.

***Note: This class may not be needed - I might be missing a simpler implementation.***

### Key Methods

- `register_function(name, function)`: Registers a function under a given name. This will never* need to be called directly as it is automatically handled by the `ToolWrapper` class.
- `get_registry()`: Returns the current registry of functions.
- `call_function(name, **kwargs)`: Calls a registered function by name, passing keyword arguments.

An example of the `FuctionRegistry` class is provided in the **Parallel Function Call** example.
