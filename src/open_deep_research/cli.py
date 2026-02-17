import asyncio
import os
import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.tree import Tree
from rich.text import Text
from rich.style import Style

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.state import AgentInputState
from open_deep_research.configuration import Configuration, SearchAPI

app = typer.Typer(help="Open Deep Research CLI")
console = Console()

@app.callback()
def main():
    """
    Open Deep Research CLI.
    """
    pass

def create_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    layout["right"].split_column(
        Layout(name="status", ratio=1),
        Layout(name="memory", ratio=1),
    )
    return layout

async def run_research(
    topic: str,
    max_depth: int,
    max_concurrent: int,
    research_model: str,
    search_api: str,
):
    """
    Async implementation of the research command.
    """
    # Configuration
    config = {
        "configurable": {
            "max_researcher_iterations": max_depth,
            "max_concurrent_research_units": max_concurrent,
            "research_model": research_model,
            "search_api": search_api,
            "allow_clarification": True,
        }
    }

    messages = [HumanMessage(content=topic)]

    while True:
        inputs = AgentInputState(messages=messages)
        final_state = {}
        root_run_id = None

        # Visualization Setup
        layout = create_layout()
        root_tree = Tree("Starting...")
        layout["left"].update(Panel(root_tree, title="Exploration Tree"))

        status_text = Text()
        layout["right"]["status"].update(Panel(status_text, title="Status Log"))

        memory_text = Text()
        layout["right"]["memory"].update(Panel(memory_text, title="Memory (Notes)"))

        layout["header"].update(Panel(f"Research Topic: [bold]{topic}[/bold]", style="bold blue"))

        run_map = {} # run_id -> Tree node

        with Live(layout, refresh_per_second=4, console=console) as live:
            async for event in deep_researcher.astream_events(inputs, config=config, version="v2"):
                name = event["name"]
                event_type = event["event"]
                run_id = event["run_id"]
                parent_run_id = event.get("parent_run_id")

                # Tree Visualization
                if event_type == "on_chain_start":
                    if not parent_run_id:
                        root_run_id = run_id
                        run_map[run_id] = root_tree
                        root_tree.label = f"[bold]{name}[/bold]"
                    elif parent_run_id in run_map:
                        parent_node = run_map[parent_run_id]
                        # Create new node
                        new_node = parent_node.add(f"[yellow]{name}[/yellow]")
                        run_map[run_id] = new_node

                elif event_type == "on_tool_start":
                    if parent_run_id in run_map:
                        parent_node = run_map[parent_run_id]
                        new_node = parent_node.add(f"[blue]Tool: {name}[/blue]")
                        run_map[run_id] = new_node

                elif event_type == "on_chain_end":
                    if run_id in run_map:
                        node = run_map[run_id]
                        # Clean label to remove markup before replacing or just append checkmark
                        # Simple way: just set style to green
                        # node.label = Text(str(node.label) + " \u2713", style="green")
                        # But label might be text or string.
                        pass # Keeping it simple for now, maybe change color

                    if run_id == root_run_id:
                        final_state = event["data"]["output"]

                    # Update Memory if notes are present
                    output = event["data"].get("output")
                    if isinstance(output, dict):
                        notes = output.get("notes")
                        if notes:
                             memory_text = Text("\n".join(notes))
                             layout["right"]["memory"].update(Panel(memory_text, title="Memory (Notes)"))

                        # Also track research brief
                        brief = output.get("research_brief")
                        if brief:
                             # Maybe prepend brief?
                             pass

                elif event_type == "on_tool_end":
                    if run_id in run_map:
                         pass

                # Update Status Log
                if event_type in ["on_chain_start", "on_tool_start"]:
                     status_text.append(f"{name} started...\n")
                     # Keep only last 10 lines
                     if len(status_text.plain.splitlines()) > 20:
                         status_text = Text("\n".join(status_text.plain.splitlines()[-20:]))
                         layout["right"]["status"].update(Panel(status_text, title="Status Log"))

        # If we didn't get a final state for some reason (e.g. error), break
        if not final_state:
            console.print("[bold red]Error: No final state returned.[/bold red]")
            break

        # Check for final report
        if final_state.get("final_report"):
            console.print(Panel(final_state["final_report"], title="Final Report", border_style="green"))
            break

        # Check for clarification
        # The last message should be the clarification question
        messages_list = final_state.get("messages", [])
        if not messages_list:
            console.print("[bold red]Error: No messages returned in final state.[/bold red]")
            break

        last_message = messages_list[-1]
        if isinstance(last_message, AIMessage):
            # Assume it's a clarification question
            # We need to ask the user
            question = last_message.content
            console.print(Panel(f"[bold yellow]Clarification needed:[/bold yellow] {question}", border_style="yellow"))

            # Get user input
            user_response = typer.prompt("Your answer")

            # Update messages with the question (already in state) and the answer
            # We take the messages from final_state which includes the AI question
            messages = messages_list + [HumanMessage(content=user_response)]

            # Loop again
        else:
            # Unexpected state
            console.print("[bold red]Unexpected state. Ending.[/bold red]")
            # console.print(final_state)
            break

@app.command()
def research(
    topic: Annotated[str, typer.Argument(help="The research topic")],
    max_depth: Annotated[int, typer.Option(help="Max researcher iterations")] = 3,
    max_concurrent: Annotated[int, typer.Option(help="Max concurrent research units")] = 3,
    research_model: Annotated[str, typer.Option(help="Model for research")] = "openai:gpt-4.1",
    search_api: Annotated[str, typer.Option(help="Search API to use")] = "tavily",
):
    """
    Run deep research on a topic.
    """
    asyncio.run(run_research(topic, max_depth, max_concurrent, research_model, search_api))

if __name__ == "__main__":
    app()
