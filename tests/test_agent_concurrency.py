import time
from types import SimpleNamespace
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from medrax.agent.agent import Agent


@tool
def slow_tool_a(x: int) -> str:
    """Tool A that takes 0.1s."""
    time.sleep(0.1)
    return f"tool_a_res_{x}"


@tool
def slow_tool_b(y: str) -> str:
    """Tool B that takes 0.1s."""
    time.sleep(0.1)
    return f"tool_b_res_{y}"


@tool
def failing_tool() -> str:
    """Tool that raises an exception."""
    raise ValueError("intentional tool failure")


class DummyModel:
    def bind_tools(self, tools):
        return self


def test_agent_parallel_tool_execution(tmp_path):
    model = DummyModel()
    tools = [slow_tool_a, slow_tool_b]
    agent = Agent(
        model=model,
        tools=tools,
        log_dir=str(tmp_path),
        log_tools=True,
        parallel_tools=True,
        max_tool_workers=2,
    )

    state = {
        "messages": [
            SimpleNamespace(
                tool_calls=[
                    {"name": "slow_tool_a", "id": "call_1", "args": {"x": 42}},
                    {"name": "slow_tool_b", "id": "call_2", "args": {"y": "chest"}},
                ]
            )
        ]
    }

    t0 = time.time()
    res = agent.execute_tools(state)
    elapsed = time.time() - t0

    messages = res["messages"]
    assert len(messages) == 2
    assert messages[0].tool_call_id == "call_1"
    assert messages[0].content == "tool_a_res_42"
    assert messages[1].tool_call_id == "call_2"
    assert messages[1].content == "tool_b_res_chest"

    # Parallel execution: 2 * 0.1s tasks run concurrently in ~0.1s - 0.15s (< 0.19s)
    assert elapsed < 0.19, f"Parallel tool execution took {elapsed:.3f}s; expected < 0.19s"


def test_agent_error_handling_in_tool(tmp_path):
    model = DummyModel()
    tools = [failing_tool]
    agent = Agent(
        model=model,
        tools=tools,
        log_dir=str(tmp_path),
        log_tools=False,
    )

    state = {
        "messages": [
            SimpleNamespace(
                tool_calls=[
                    {"name": "failing_tool", "id": "call_fail", "args": {}},
                    {"name": "non_existent_tool", "id": "call_invalid", "args": {}},
                ]
            )
        ]
    }

    res = agent.execute_tools(state)
    messages = res["messages"]
    assert len(messages) == 2
    assert "Error executing tool failing_tool" in messages[0].content
    assert "invalid tool" in messages[1].content

