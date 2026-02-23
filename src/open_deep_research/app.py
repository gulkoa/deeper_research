import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from open_deep_research.deep_researcher import deep_researcher

@cl.on_chat_start
def start():
    """Initialize the session."""
    cl.user_session.set("messages", [])

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages."""
    messages = cl.user_session.get("messages")
    messages.append(HumanMessage(content=message.content))

    # Initialize the message for the response
    msg = cl.Message(content="")
    await msg.send()

    # Create a step to show progress
    async with cl.Step(name="Open Deep Research") as step:
        step.input = "Initializing..."
        await step.send()

        # Configuration
        config = {"configurable": {"thread_id": message.id}}
        inputs = {"messages": messages}

        final_report_text = ""
        streamed_text = False
        final_state = None

        # Stream events from the graph
        async for event in deep_researcher.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            name = event["name"]
            data = event["data"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Capture the final state from the root graph completion
            if kind == "on_chain_end" and name == "LangGraph":
                final_state = data.get("output")

            # Stream tokens from the final report generation
            if kind == "on_chat_model_stream":
                if node == "final_report_generation":
                    content = data["chunk"].content
                    if content:
                        await msg.stream_token(content)
                        final_report_text += content
                        streamed_text = True

                # Provide feedback for other nodes
                elif node == "clarify_with_user":
                    step.input = "Analyzing request..."
                    await step.update()
                elif node == "write_research_brief":
                    step.input = "Writing research brief..."
                    await step.update()
                elif node == "research_supervisor":
                    step.input = "Conducting research (this may take a while)..."
                    await step.update()

            # Handle node completions for progress updates
            elif kind == "on_chain_end":
                if node == "write_research_brief":
                    step.input = "Research Brief Completed."
                    await step.update()
                elif node == "research_supervisor":
                    step.input = "Research Phase Completed."
                    await step.update()
                elif node == "compress_research":
                    step.input = "Synthesizing findings..."
                    await step.update()

        # If we didn't stream the report (e.g., it was a clarification question),
        # retrieve the last message from the final state and send it.
        if not streamed_text and final_state and "messages" in final_state:
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
                if content:
                    await msg.stream_token(content)
                    final_report_text = content

    # Finalize the message
    await msg.update()

    # Update history with the AI response
    if final_report_text:
        messages.append(AIMessage(content=final_report_text))
        cl.user_session.set("messages", messages)
