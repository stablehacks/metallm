import chainlit as cl

@cl.on_message
async def on_message(message: str):
    return message  # This is correct, no await needed here

# Run the app
if __name__ == "__main__":
    cl.run()

