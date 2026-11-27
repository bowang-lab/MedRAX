"""Tool-call errors are returned to the model instead of ending the run.

Uses a scripted chat model, so no model weights, API keys or network are needed.
"""

import ast
import importlib.util
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from medrax.agent.agent import Agent

# Load the image visualizer by file path: importing medrax.tools would also load
# the model-backed tools (torch, model weights), which these tests don't need.
_UTILS = Path(__file__).resolve().parents[1] / "medrax" / "tools" / "utils.py"
_spec = importlib.util.spec_from_file_location("medrax_tools_utils", _UTILS)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
ImageVisualizerTool = _utils.ImageVisualizerTool


class ScriptedModel(BaseChatModel):
    """Returns the scripted replies in order and records what it was sent."""

    replies: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _agent(replies: list[AIMessage], **kwargs: Any):
    model = ScriptedModel(replies=replies, seen=[])
    agent = Agent(model, [ImageVisualizerTool()], log_dir=tempfile.mkdtemp(), **kwargs)
    return agent, model


def test_invalid_tool_arguments_are_returned_to_the_model():
    agent, model = _agent(
        [
            _tool_call(
                "image_visualizer", {"path": "/data/xray.png"}
            ),  # wrong argument name
            AIMessage(content="I could not display the image."),
        ]
    )
    out = agent.workflow.invoke({"messages": [HumanMessage("Show the X-ray")]})

    assert out["messages"][-1].content == "I could not display the image."
    tool_message = next(m for m in model.seen[1] if isinstance(m, ToolMessage))
    assert "ValidationError" in tool_message.content
    assert "image_path" in tool_message.content


def test_thread_has_no_unanswered_tool_call_after_invalid_arguments():
    agent, model = _agent(
        [
            _tool_call("image_visualizer", {"path": "/data/xray.png"}),
            AIMessage(content="I could not display the image."),
            AIMessage(content="Hello again."),
        ],
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t"}}
    agent.workflow.invoke({"messages": [HumanMessage("Show the X-ray")]}, cfg)
    agent.workflow.invoke({"messages": [HumanMessage("Are you there?")]}, cfg)

    history = model.seen[-1]
    called = {
        c["id"] for m in history if isinstance(m, AIMessage) for c in m.tool_calls
    }
    answered = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    assert called and called <= answered


def test_one_failing_call_does_not_affect_another_in_the_same_response(tmp_path):
    image = tmp_path / "xray.png"
    image.write_bytes(b"not really a png; the visualizer only checks the path exists")
    agent, model = _agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "image_visualizer",
                        "args": {"path": "x.png"},
                        "id": "bad",
                    },
                    {
                        "name": "image_visualizer",
                        "args": {"image_path": str(image)},
                        "id": "good",
                    },
                ],
            ),
            AIMessage(content="Shown."),
        ]
    )
    agent.workflow.invoke({"messages": [HumanMessage("Show the X-ray")]})

    results = {m.tool_call_id: m for m in model.seen[1] if isinstance(m, ToolMessage)}
    assert set(results) == {"bad", "good"}
    bad_output, _ = ast.literal_eval(results["bad"].content)
    good_output, _ = ast.literal_eval(results["good"].content)
    assert "ValidationError" in bad_output["error"]
    assert good_output == {"image_path": str(image)}


def test_unknown_tool_result_can_be_rendered_by_the_interface():
    agent, _model = _agent(
        [
            _tool_call("image_viewer", {"image_path": "/data/xray.png"}),
            AIMessage(content="Done."),
        ]
    )
    out = agent.workflow.invoke({"messages": [HumanMessage("Show the X-ray")]})

    tool_message = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    # interface.py renders tool results with eval(message.content)[0]; the result
    # must be a Python literal of the (output, metadata) shape the tools return.
    output, metadata = ast.literal_eval(tool_message.content)
    assert "Unknown tool: image_viewer" in output["error"]
    assert metadata["tool_call_status"] == "failed"
