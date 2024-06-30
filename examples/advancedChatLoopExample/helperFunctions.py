import json
import asyncio
from rich import print
from llmFunctionDecorator import FunctionRegistry
from litellm import acompletion


def append_to_message_history(messages, response_message):
    """Appends a new response message to the messages history and logs the addition."""
    messages.append(response_message)
    print("[red]     Message appended to history.[/red]")


async def chat_completion(messages):
    """Retrieve response from LLM based on the provided messages and manage tool calls."""

    response = await acompletion(
        model='gpt-4o',
        messages=messages,
        tools=FunctionRegistry.tools(),
        tool_choice=FunctionRegistry.tool_choice(),
        max_tokens=1000,
        stream=True
    )

    contents = ""           # Initialize a variable to store the response contents
    tool_calls_dict = {}    # Temporary dictionary to store tool calls by index

    async def text_iterator():
        """Helper function to iterate over the streamed response."""
        nonlocal contents
        nonlocal tool_calls_dict

        # Iterate over chunks of the response as they arrive
        async for chunk in response:
            # Get the delta part of the current chunk, which contains updates
            delta = chunk.choices[0].delta

            # If the delta contains tool calls, process them
            if "tool_calls" in delta and delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    index = tool_call.index  # Get the index of the tool call
                    if index not in tool_calls_dict:  # Create a new dictionary entry if one doesn't exist
                        tool_calls_dict[index] = {
                            "id": tool_call.id,
                            "name": None,
                            "arguments": ""
                        }

                    # If the tool call has a function name, add it to the dictionary
                    if tool_call.function.name is not None:
                        tool_calls_dict[index]["name"] = tool_call.function.name

                    # If the tool call has function arguments, append them to the existing arguments
                    if tool_call.function.arguments is not None:
                        tool_calls_dict[index]["arguments"] += tool_call.function.arguments

            # If the delta contains new content, process it
            if delta.content is not None:
                print(delta.content, end='')  # Print the content as it's being streamed
                contents += delta.content  # Add the new content to the total contents
                yield delta.content  # Yield the content so it can be processed by the caller

    # Process the streamed response
    async for _ in text_iterator():
        pass

    print("\n", end='')

    response_message = {"role": "assistant", "content": contents}

    # Convert the tool_calls_dict into a list
    tool_calls = list(tool_calls_dict.values())

    if tool_calls:  # If there are any tool calls
        # Transform each tool call into the expected format
        transformed_tool_calls = [{
            "id": call["id"],
            "function": {"name": call["name"], "arguments": call["arguments"]},
            "type": "function"
        } for call in tool_calls]

        response_message["tool_calls"] = transformed_tool_calls  # Add the list of tool calls to the response message

    return response_message


async def async_tool_call(function_name, function_args):
    """
    Calls a tool function asynchronously with provided arguments.

    Parameters:
    - function_name (str): The name of the tool function to call.
    - function_args (dict): The arguments to pass to the tool function.

    Returns:
    - dict: The result of the tool function call.
    """
    # Call the tool function with the provided arguments
    result = FunctionRegistry.call_function(function_name, **function_args)

    return result


async def handle_tool_calls(response_message, messages):
    """
    Handles tool calls in the response message and updates the message history.

    Parameters:
    - response_message (dict): The response message containing tool calls.
    - messages (list): The list of messages in the conversation history.

    Returns:
    - dict: The new response message after handling tool calls.
    """

    # Create a list of tasks by calling each tool function asynchronously
    tasks = [
        asyncio.create_task(
            async_tool_call(
                tool_call['function']['name'],
                json.loads(tool_call['function']['arguments'])
            )
        ) for tool_call in response_message['tool_calls']
    ]

    # Wait for all tasks to complete
    function_responses = await asyncio.gather(*tasks)

    # Iterate over each tool call and its corresponding function response
    for tool_call, function_response in zip(response_message['tool_calls'], function_responses):
        # Print the function response
        print(
            f"     Function response for {tool_call['function']['name']}({json.loads(tool_call['function']['arguments'])}):\n     ",
            function_response
        )

        # Append the function response to the message history
        append_to_message_history(messages, {
            "tool_call_id": tool_call['id'],
            "role": "tool",
            "name": tool_call['function']['name'],
            "content": function_response,
        })

    # Print the assistant's response label
    print(f'[green]assistant: [/green]', end='')

    # Make a second completion call to OpenAI with the updated message history
    second_response_message = await chat_completion(messages)

    # Append the response from the second completion call to the message history
    append_to_message_history(messages, second_response_message)

    return second_response_message