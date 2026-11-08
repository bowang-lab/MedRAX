import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from .ledger import FindingsLedger, build_envelope, extract_confidence

_ = load_dotenv()

# Whether the installed langchain ToolMessage exposes the `artifact` field
# (SPEC-LEDGER-001 CD-1). If absent, the envelope rides in `additional_kwargs`.
_TOOLMESSAGE_HAS_ARTIFACT = "artifact" in getattr(ToolMessage, "model_fields", {})


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Findings ledger sidecar (SPEC-LEDGER-001). Always available for
        # retrieval; writes are gated by log_tools inside _save_tool_calls.
        self.ledger = FindingsLedger(log_dir=log_dir or "logs")

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(
        self, state: AgentState, config: Optional[RunnableConfig] = None
    ) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.
            config (Optional[Dict[str, Any]]): The LangGraph runnable config.
                LangGraph injects it by parameter name; it carries the
                ``thread_id`` used to key the findings ledger (SPEC-LEDGER-001).

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        thread_id = None
        if config:
            thread_id = (config.get("configurable") or {}).get("thread_id")

        results: List[ToolMessage] = []
        envelopes: List[Dict[str, Any]] = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            name, call_id, call_args = call["name"], call["id"], call["args"]
            raised = False
            if name not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                try:
                    result = self.tools[name].invoke(call_args)
                except Exception as exc:  # capture, do not abort the run (REQ-8)
                    print(f"\n....tool raised: {exc}....")
                    result = ({"error": str(exc)}, {"analysis_status": "failed"})
                    raised = True

            # Build the provenance envelope (SPEC-LEDGER-001). Additive: the
            # ToolMessage.content below stays the byte-identical str(result).
            envelope = build_envelope(
                name,
                call_id,
                call_args,
                result,
                raised=raised,
                confidence=extract_confidence(result),
            )
            envelopes.append(envelope)

            message_kwargs: Dict[str, Any] = {
                "tool_call_id": call_id,
                "name": name,
                "args": call_args,
                "content": str(result),
            }
            if _TOOLMESSAGE_HAS_ARTIFACT:
                message_kwargs["artifact"] = envelope
            else:
                message_kwargs["additional_kwargs"] = {"finding_envelope": envelope}
            results.append(ToolMessage(**message_kwargs))

        self._save_tool_calls(results, envelopes, thread_id)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(
        self,
        tool_calls: List[ToolMessage],
        envelopes: Optional[List[Dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """
        Save tool calls to a JSON file and append envelopes to the findings ledger.

        When ``log_tools`` is False, both the per-batch log file and the ledger
        write are skipped (SPEC-LEDGER-001 REQ-9).

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
            envelopes (Optional[List[Dict[str, Any]]]): Provenance envelopes,
                one per tool call, to append to the ledger (REQ-6).
            thread_id (Optional[str]): Conversation key for the ledger (REQ-3).
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

        # Findings ledger sidecar (REQ-6): persist each envelope keyed by
        # thread_id, alongside the per-batch log written above.
        for envelope in envelopes or []:
            self.ledger.append(thread_id, envelope)
