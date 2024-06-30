"""
An advanced example of llmFunctionDecorator usage in a chat application. This example contains a chat loop,
response streaming, asynchronous parallel tool calls, and a summarization of the tool call results.
"""


import os
import asyncio
from rich import print

import tools
from llmFunctionDecorator import FunctionRegistry
from helperFunctions import append_to_message_history, handle_tool_calls, chat_completion

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY HERE"


async def chat_loop():
    """
    Main chat loop that handles user input and gathers responses.
    """

    # Display callable functions and statuses from FunctionRegistry
    print("\nCallable functions present in FunctionRegistry:", FunctionRegistry.get_registry())
    print("\nAll functions in FunctionRegistry:\n", FunctionRegistry.registry_status(), "\n")

    # Initialize the system prompt for the assistant
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant. You do not need to make tool calls with every response."
            "If the question was previously answered by you, just look in the message history."
        )
    }

    # Initialize the message history with the system prompt
    messages = [system_prompt]

    try:
        while True:
            # Get user input
            user_input = {"role": "user", "content": str(input("user: "))}
            append_to_message_history(messages, user_input)

            # Print the assistant's response label
            print(f'[green]assistant: [/green]', end='')

            # Get chat completion from the assistant
            response_message = await chat_completion(messages)
            # Append assistant's response to message history
            append_to_message_history(messages, response_message)

            # Handle tool calls if present in the response
            if 'tool_calls' in response_message and response_message['tool_calls']:
                await handle_tool_calls(response_message, messages)
            else:
                print("[yellow]     No tool calls in this response.[/yellow]")
                continue

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(chat_loop())
