"""
llmFunctionDecorator
by A-M-D-R-3-W on GitHub

A Python module designed to simplify the process of creating and managing
function calls to OpenAI's API, as well as models using LiteLLM's API framework.
"""

import inspect
from functools import wraps

class FunctionRegistry:
    """
    A registry for storing and managing tool instances.
    Enables registration, retrieval, and invocation of functions by name,
    along with their enabled status.
    """
    _registry = {}

    @classmethod
    def register_function(cls, name, tool_instance):
        """
        Registers a ToolWrapper instance with a given name.
        :param name: The name of the function to register.
        :param tool_instance: The ToolWrapper instance to register.
        """
        cls._registry[name] = {
            "instance": tool_instance,
            "enabled": tool_instance.enabled
        }

    @classmethod
    def get_registry(cls):
        """
        Returns a dictionary of all enabled function instances registered.
        :return: Dictionary of function name to ToolWrapper instance for enabled functions.
        """
        enabled_functions = {}
        for name, info in cls._registry.items():
            if info["enabled"]:
                enabled_functions[name] = info["instance"]
        return enabled_functions

    @classmethod
    def registry_status(cls):
        """
        Returns a string summary of all registered functions and their enabled status.
        :return: Formatted string listing function names and their enabled status.
        """
        registry_entries = []
        for name, info in cls._registry.items():
            registry_entries.append(f"Function: {name}, Enabled: {info['enabled']}")
        # Join the list into a single string with newline separators
        return "\n".join(registry_entries)

    @classmethod
    def tools(cls):
        """
        Retrieve registered ToolWrapper instances and convert to their dictionary representations.
        This is what is sent to the model as the tools parameter.
        :return: List of dictionaries representing each registered ToolWrapper instance. None if no tools are registered.
        """
        # Retrieve registered ToolWrapper instances directly
        registered_tools = cls.get_registry().values()
        # Convert each ToolWrapper instance to its dictionary representation
        tools = [tool.to_dict() for tool in registered_tools]
        return tools if tools else None

    @classmethod
    def tool_choice(cls, function_ref=None):
        """
        Generates a tool choice specification based on the function reference provided,
        or the state of the function registry if no function reference is provided.
        - Returns "auto" if the registry has one or more enabled functions and no function_ref is provided.
        - Returns None if the registry is empty and no function_ref is provided.
        - Ensures the function is both registered and enabled when a reference is provided.

        :param function_ref: Reference to the function for which tool choice specification is generated. Optional.
        :return: "auto", None, or a dictionary specifying the function for tool choice.
        :raises: ValueError if function_ref is provided but not callable, not registered, or not enabled.
        """
        if function_ref is None:
            # Check if registry has enabled functions; decide "auto" or None
            return "auto" if any(info["enabled"] for info in cls._registry.values()) else None
        else:
            if not callable(function_ref):
                raise ValueError("The provided function_ref is not callable. Please provide a valid function.")

            function_name = function_ref.__name__
            # Check if the function is registered and enabled
            if function_name in cls._registry and cls._registry[function_name]["enabled"]:
                return {
                    "type": "function",
                    "function": {
                        "name": function_name
                    }
                }
            else:
                raise ValueError(
                    f"The function '{function_name}' is either not registered in the FunctionRegistry or is disabled.")

    @classmethod
    def call_function(cls, name, **kwargs):
        """
        Attempts to call a registered function by name with provided arguments.
        Handles and returns meaningful messages for unregistered or disabled functions and argument binding issues.
        :param name: The name of the function to call.
        :param kwargs: Arguments to pass to the function call.
        :return: Result of the function call or error message.
        """
        if name not in cls._registry:
            return f"Function {name} is not registered and cannot be called."

        tool_info = cls._registry[name]
        if not tool_info["enabled"]:
            return f"Function {name} is currently disabled and cannot be called."

        tool_instance = tool_info["instance"]
        func = tool_instance.function_ref
        sig = inspect.signature(func)  # Get function signature
        try:
            bound_args = sig.bind(**kwargs)  # Bind given arguments
        except TypeError as e:
            return f"Error binding arguments for {name}: {e}"
        bound_args.apply_defaults()  # Apply defaults for missing args

        return func(*bound_args.args, **bound_args.kwargs)


class ToolWrapper:
    """
    A wrapper for functions providing additional metadata and utility methods.
    Registers itself upon instantiation.
    """

    def __init__(self, purpose, required=None, function_ref=None, enabled=True, **kwargs):
        """
        Initializes a ToolWrapper instance with function reference, purpose, parameter info, and metadata.
        Automatically registers the function in the FunctionRegistry.
        :param purpose: Purpose or description of the function.
        :param required: List of required parameter names.
        :param function_ref: Reference to the actual function to wrap.
        :param enabled: Flag indicating if the function is initially enabled.
        :param kwargs: Additional keyword arguments for parameter types and descriptions.
        """
        if function_ref is None:
            raise ValueError("'function_ref' must be provided")
        if not callable(function_ref):
            raise TypeError("'function_ref' must be callable")

        self.name = function_ref.__name__       # Function name
        self.function_ref = function_ref        # Reference to the actual function
        self.enabled = enabled                  # Store the provided enabled value or default to True
        self.purpose = purpose.strip()          # Purpose/description of the function
        self.parameters = {}                    # Dictionary to hold parameter info
        self.required = required or []          # List of required parameter names

        # Inspect the function's parameters
        func_signature = inspect.signature(function_ref).parameters

        # Create a set of parameter names for which descriptions are expected to be provided
        param_keys = {key for key in kwargs if not key.endswith('_description')}
        # Create a set of parameter descriptions, stripping '_description' from the keys
        description_keys = {key.replace('_description', '') for key in kwargs if key.endswith('_description')}

        # Verify each parameter against the function's signature
        for param_name in param_keys:
            if param_name not in func_signature:
                raise ValueError(
                    f"Parameter '{param_name}' doesn't exist in the function '{function_ref.__name__}'")

        # Check for descriptions provided without the corresponding parameter
        for desc_key in description_keys:
            if desc_key not in param_keys:
                raise ValueError(f"Description provided for '{desc_key}', but '{desc_key}' parameter is not present")

        # Check for each parameter having a corresponding description
        for param_name in param_keys:
            if param_name not in description_keys:
                raise ValueError(f"Description for '{param_name}' not provided")
            # Add parameter information using the _add_parameter method
            self._add_parameter(param_name, kwargs[param_name])

        # Add parameter descriptions
        for param_name, param_description in kwargs.items():
            if param_name.endswith('_description'):
                true_param_name = param_name.replace('_description', '')
                if true_param_name in self.parameters:
                    self.parameters[true_param_name]['description'] = param_description

        # Ensure all required parameters are amongst those defined
        for req_param in self.required:
            if req_param not in self.parameters:
                raise ValueError(f"Required parameter '{req_param}' is not defined among parameters")

    def _add_parameter(self, name, param_info):
        """
        Adds a parameter to the internal parameters dictionary.
        Handles type mapping and enumerations.
        :param name: Name of the parameter.
        :param param_info: Type of the parameter or list of enumeration values.
        """
        type_mapping = {  # Map Python types to JSON schema types.
            int: 'integer',
            float: 'number',
            str: 'string',
            bool: 'boolean',
            list: "array",
            tuple: "array",
            dict: "object",
            None: "null",
        }

        # Check if the parameter is an enumeration
        if isinstance(param_info, list):
            parameter_type = 'array'
            enum_values = param_info
            # Check if the enum list is not empty
            if not enum_values:
                # If it is empty, throw an error
                raise ValueError(f"Enum for parameter '{name}' must have at least one value.")
        else:  # Otherwise, use the type mapping
            parameter_type = type_mapping.get(param_info, 'string')
            enum_values = None

        # Construct the parameter dictionary
        parameter = {
            'name': name,
            'type': parameter_type,
            'enum': enum_values if enum_values else None
        }

        # Only add the parameter if it's not an empty enum
        if not enum_values or (enum_values and len(enum_values) > 0):
            self.parameters[name] = parameter

    def to_dict(self):
        """
        Converts the ToolWrapper instance into a standard dictionary format.
        :return: Dictionary representation of the ToolWrapper instance.
        """
        properties = {}
        required_fields = []

        # Convert parameters to JSON schema properties
        for param_name, param_info in self.parameters.items():
            props = {'type': param_info['type']}
            if param_info.get('enum'):  # Add enum values if present
                props['enum'] = param_info['enum']
            if param_info.get('description'):  # Add description if present
                props['description'] = param_info['description']
            if param_name in self.required:  # Mark as required if necessary
                required_fields.append(param_name)

            properties[param_name] = props

        # Construct and return the full function metadata dictionary
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.purpose,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_fields
                }
            }
        }


def tool(purpose, required=None, enabled=True, **kwargs):
    """
    A decorator for registering functions with additional metadata.
    Wraps the function in a ToolWrapper instance and registers it.
    :param purpose: Purpose or description of the function.
    :param required: List of required parameter names.
    :param enabled: Flag indicating if the function is initially enabled.
    :param kwargs: Additional keyword arguments for parameter types and descriptions.
    :return: Decorated function wrapped for registration and metadata association.
    """
    def decorator(func):
        # Create a ToolWrapper instance around the function
        tool_wrapper_instance = ToolWrapper(
            function_ref=func,
            purpose=purpose,
            required=required,
            enabled=enabled,
            **kwargs,
        )

        # Register the ToolWrapper instance with the FunctionRegistry
        FunctionRegistry.register_function(func.__name__, tool_wrapper_instance)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    return decorator
