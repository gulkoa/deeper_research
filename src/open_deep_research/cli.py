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
from rich.theme import Theme
from rich.markdown import Markdown

from langchain_core.messages import HumanMessage, AIMessage

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.state import AgentInputState
from open_deep_research.configuration import Configuration, SearchAPI

# Set up a nice theme
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "title": "bold blue",
    "header": "bold white on blue",
})
console = Console(theme=custom_theme)

app = typer.Typer(help="Open Deep Research CLI")

@app.callback()
def main():
    """
    Open Deep Research CLI.
    """
    pass

class ResearchMonitor:
    def __init__(self, topic: str, config: dict):
        self.topic = topic
        self.config = config
        self.tree = Tree(f"[bold blue]Root: {topic}[/bold blue]")
        self.logs = []
        self.findings = []
        self.brief = "Initializing..."
        self.run_map = {} # run_id -> Tree node
        self.layout = self._create_layout()

        # Initial population of layout
        self.layout["left"].update(Panel(self.tree, title="Execution Graph", border_style="blue"))
        self.layout["logs"].update(Panel(Text("Waiting for events..."), title="Live Logs", border_style="cyan"))
        self.layout["brief"].update(Panel(Markdown(self.brief), title="Research Brief", border_style="yellow"))
        self.layout["findings"].update(Panel(Text("No findings yet."), title="Key Findings", border_style="green"))

    def _create_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
        )
        layout["header"].update(Panel(f"Research Topic: [bold]{self.topic}[/bold]", style="bold white on blue"))

        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )

        layout["right"].split_column(
            Layout(name="brief", size=10),
            Layout(name="logs", ratio=1),
            Layout(name="findings", ratio=1),
        )
        return layout

    def update_logs(self, message: str, style: str = ""):
        self.logs.append(Text(message, style=style))
        if len(self.logs) > 20:
            self.logs = self.logs[-20:]

        log_text = Text()
        for log in self.logs:
            log_text.append(log)
            log_text.append("\n")

        self.layout["logs"].update(Panel(log_text, title="Live Logs", border_style="cyan"))

    def update_findings(self, notes: list[str]):
        if not notes:
            return

        # Deduplicate findings
        existing_findings = set(self.findings)
        new_findings = [n for n in notes if n not in existing_findings]

        if new_findings:
            self.findings.extend(new_findings)
            # Limit display to last 10 findings to avoid overflow
            display_findings = self.findings[-10:]
            formatted_findings = "\n".join([f"• {f[:100]}..." if len(f) > 100 else f"• {f}" for f in display_findings])
            self.layout["findings"].update(Panel(formatted_findings, title=f"Key Findings ({len(self.findings)})", border_style="green"))

    def update_brief(self, brief: str):
        if brief and brief != self.brief:
            self.brief = brief
            self.layout["brief"].update(Panel(Markdown(brief), title="Research Brief", border_style="yellow"))

    def process_event(self, event: dict):
        name = event["name"]
        event_type = event["event"]
        run_id = event["run_id"]
        parent_run_id = event.get("parent_run_id")
        data = event.get("data", {})

        # -- Tree Updates --
        if event_type == "on_chain_start":
            if not parent_run_id:
                # Root run
                self.run_map[run_id] = self.tree
                self.tree.label = f"[bold blue]Research: {self.topic}[/bold blue]"
            elif parent_run_id in self.run_map:
                parent_node = self.run_map[parent_run_id]
                # Identify node type
                icon = "⛓️"
                style = "white"

                if name == "supervisor":
                    icon = "👨‍💼"
                    style = "bold yellow"
                elif name == "researcher":
                    icon = "🕵️"
                    style = "bold magenta"
                elif name == "clarify_with_user":
                    icon = "❓"
                    style = "bold cyan"
                elif name == "write_research_brief":
                    icon = "📝"
                    style = "bold yellow"
                elif name == "final_report_generation":
                    icon = "📄"
                    style = "bold green"

                node_label = f"{icon} {name}"
                new_node = parent_node.add(f"[{style}]{node_label}[/{style}]")
                self.run_map[run_id] = new_node
                self.update_logs(f"Started: {name}", style="dim")

        elif event_type == "on_tool_start":
            if parent_run_id in self.run_map:
                parent_node = self.run_map[parent_run_id]
                icon = "🔧"
                style = "cyan"
                if name == "tavily_search":
                    icon = "🔎"
                elif name == "think_tool":
                    icon = "🧠"

                new_node = parent_node.add(f"[{style}]{icon} {name}[/{style}]")
                self.run_map[run_id] = new_node
                self.update_logs(f"Tool Start: {name}", style="dim cyan")

        elif event_type == "on_chain_end":
            if run_id in self.run_map:
                node = self.run_map[run_id]
                # Mark as done
                if not str(node.label).endswith("✅"):
                     node.label = Text.from_markup(f"{node.label} ✅")

            # Check for outputs
            output = data.get("output")
            if isinstance(output, dict):
                if "research_brief" in output:
                    self.update_brief(output["research_brief"])
                if "notes" in output and output["notes"]:
                     self.update_findings(output["notes"])

            self.update_logs(f"Finished: {name}", style="dim green")

        elif event_type == "on_tool_end":
            if run_id in self.run_map:
                node = self.run_map[run_id]
                # Mark as done
                if not str(node.label).endswith("✅"):
                     node.label = Text.from_markup(f"{node.label} ✅")
            self.update_logs(f"Tool End: {name}", style="dim green")

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
    monitor = ResearchMonitor(topic, config)

    while True:
        inputs = AgentInputState(messages=messages)
        final_state = {}

        with Live(monitor.layout, refresh_per_second=4, console=console) as live:
            monitor.update_logs("Starting research process...", style="bold blue")

            async for event in deep_researcher.astream_events(inputs, config=config, version="v2"):
                monitor.process_event(event)

                # Check for final output in the stream if available
                # Note: 'on_chain_end' for the root run is the best place to catch final state
                if event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                    final_state = event["data"]["output"]

        # If we didn't get a final state for some reason
        if not final_state:
            console.print("[bold red]Error: No final state returned.[/bold red]")
            break

        # Check for final report
        if final_state.get("final_report"):
            console.print(Panel(Markdown(final_state["final_report"]), title="Final Report", border_style="bold green"))
            break

        # Check for clarification
        messages_list = final_state.get("messages", [])
        if not messages_list:
            console.print("[bold red]Error: No messages returned in final state.[/bold red]")
            break

        last_message = messages_list[-1]

        # Check if clarification is needed (AIMessage at the end and no final report)
        if isinstance(last_message, AIMessage):
            # The last message is the clarification question
            question = last_message.content
            console.print(Panel(f"[bold yellow]Clarification needed:[/bold yellow] {question}", border_style="yellow"))

            # Get user input
            user_response = typer.prompt("Your answer")

            # Update messages with the question (already in state) and the answer
            messages = messages_list + [HumanMessage(content=user_response)]

            monitor.update_logs("Received user input. Resuming...", style="bold blue")

        else:
            # Unexpected state
            console.print("[bold red]Unexpected state. Ending.[/bold red]")
            break

@app.command()
def research(
    topic: Annotated[str, typer.Argument(help="The research topic")],
    max_depth: Annotated[int, typer.Option(help="Max researcher iterations")] = 3,
    max_concurrent: Annotated[int, typer.Option(help="Max concurrent research units")] = 3,
    research_model: Annotated[str, typer.Option(help="Model for research")] = "openai:gpt-4o",
    search_api: Annotated[str, typer.Option(help="Search API to use")] = "tavily",
):
    """
    Run deep research on a topic.
    """
    asyncio.run(run_research(topic, max_depth, max_concurrent, research_model, search_api))

if __name__ == "__main__":
    app()
