"""Oracle tests for SPEC-LEDGER-001 — findings ledger sidecar (AC-1..AC-8).

Stub-tool based: no GPU or model weights required. Each test drives
``Agent.execute_tools`` directly with a state carrying an ``AIMessage``'s
``tool_calls`` and a config carrying ``thread_id``. Maps 1:1 to acceptance.md.
"""

import ast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from medrax.agent.agent import Agent


class _StubTool(BaseTool):
    """Deterministic stand-in for a CXR tool. ``behavior`` is ('return', value)
    or ('raise', exception)."""

    name: str = "stub"
    description: str = "stub tool for tests"
    behavior: tuple = ("return", None)

    def _run(self, image_path: str = "", **kwargs):
        mode, payload = self.behavior
        if mode == "raise":
            raise payload
        return payload


class _FakeModel:
    """Stands in for the LLM; only ``bind_tools`` is used during Agent init."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):  # pragma: no cover - not exercised here
        raise NotImplementedError


class _NonSerializable:
    """A value that json cannot encode directly (forces string coercion)."""

    def __repr__(self):
        return "<NonSerializable>"


def _tool(name, behavior):
    return _StubTool(name=name, description=f"{name} stub", behavior=behavior)


def _agent(tools, tmp_path, log_tools=True):
    return Agent(model=_FakeModel(), tools=tools, log_tools=log_tools, log_dir=str(tmp_path))


def _call(tool_name, call_id, image_path="a.png"):
    return {"name": tool_name, "args": {"image_path": image_path}, "id": call_id, "type": "tool_call"}


def _state(calls):
    return {"messages": [AIMessage(content="", tool_calls=calls)]}


def _cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _completed(payload):
    return ("return", (payload, {"analysis_status": "completed"}))


# --- AC-1: cardinality across 2 tools + unchanged graph topology --------------
def test_ac1_cardinality_and_topology(tmp_path):
    agent = _agent(
        [
            _tool("chest_xray_classifier", _completed({"Pneumonia": 0.5})),
            _tool("chest_xray_segmentation", _completed({"lung": 1})),
        ],
        tmp_path,
    )
    calls = [
        _call("chest_xray_classifier", "call-1", "a.png"),
        _call("chest_xray_segmentation", "call-2", "a.png"),
        _call("chest_xray_classifier", "call-3", "b.png"),
    ]
    agent.execute_tools(_state(calls), _cfg("t-abc"))

    entries = agent.ledger.get("t-abc")
    assert len(entries) == 3
    assert {e["tool_call_id"] for e in entries} == {"call-1", "call-2", "call-3"}
    assert [e["tool_name"] for e in entries] == [
        "chest_xray_classifier",
        "chest_xray_segmentation",
        "chest_xray_classifier",
    ]

    graph = agent.workflow.get_graph()
    nodes = set(graph.nodes.keys())
    assert {"__start__", "process", "execute", "__end__"} <= nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "process") in edges
    assert ("execute", "process") in edges
    assert ("process", "execute") in edges
    assert ("process", "__end__") in edges


# --- AC-2: repeated same-tool calls are not deduplicated ----------------------
def test_ac2_repeated_calls_not_deduplicated(tmp_path):
    agent = _agent([_tool("chest_xray_classifier", _completed({"x": 0.1}))], tmp_path)
    calls = [
        _call("chest_xray_classifier", "c1", "same.png"),
        _call("chest_xray_classifier", "c2", "same.png"),
    ]
    agent.execute_tools(_state(calls), _cfg("t-rep"))

    entries = agent.ledger.get("t-rep")
    assert len(entries) == 2
    assert {e["tool_call_id"] for e in entries} == {"c1", "c2"}


# --- AC-3: envelope field completeness + provenance + confidence --------------
def test_ac3_envelope_fields(tmp_path):
    agent = _agent([_tool("chest_xray_classifier", _completed({"Pneumonia": 0.81}))], tmp_path)
    agent.execute_tools(_state([_call("chest_xray_classifier", "call-x")]), _cfg("t-fields"))

    entry = agent.ledger.get("t-fields")[0]
    for field in ("tool_name", "tool_call_id", "args", "raw_output", "provenance_source", "timestamp"):
        assert field in entry and entry[field] is not None
    assert entry["tool_name"] == "chest_xray_classifier"
    assert entry["provenance_source"] == "DenseNet-121"
    assert entry["provenance_source"] != entry["tool_name"]
    assert entry["tool_call_id"] == "call-x"
    assert entry["args"] == {"image_path": "a.png"}
    assert entry["raw_output"][0] == {"Pneumonia": 0.81}
    assert "confidence" in entry  # a confidence signal was available


# --- AC-4: content stays str(result) and eval-parseable; envelope on artifact -
def test_ac4_backward_compat_content(tmp_path):
    result_value = ({"Pneumonia": 0.81}, {"analysis_status": "completed"})
    agent = _agent([_tool("chest_xray_classifier", ("return", result_value))], tmp_path)
    out = agent.execute_tools(_state([_call("chest_xray_classifier", "call-x")]), _cfg("t-bc"))

    msg = out["messages"][0]
    assert msg.content == str(result_value)
    assert ast.literal_eval(msg.content)[0] == {"Pneumonia": 0.81}
    assert msg.artifact is not None and msg.artifact["tool_call_id"] == "call-x"


# --- AC-5: ledger keyed and retrievable by thread_id --------------------------
def test_ac5_thread_id_keying(tmp_path):
    agent = _agent([_tool("chest_xray_classifier", _completed({"x": 0.1}))], tmp_path)
    agent.execute_tools(
        _state([_call("chest_xray_classifier", "a1"), _call("chest_xray_classifier", "a2")]),
        _cfg("t-1"),
    )
    agent.execute_tools(_state([_call("chest_xray_classifier", "b1")]), _cfg("t-2"))

    assert len(agent.ledger.get("t-1")) == 2
    assert len(agent.ledger.get("t-2")) == 1
    assert {e["tool_call_id"] for e in agent.ledger.get("t-1")} == {"a1", "a2"}
    assert {e["tool_call_id"] for e in agent.ledger.get("t-2")} == {"b1"}


# --- AC-6: tool returns error tuple / non-serializable / raises ---------------
def test_ac6_error_and_raise_recovery(tmp_path):
    agent = _agent(
        [
            _tool("err1", ("return", ({"error": "file not found"}, {"analysis_status": "failed"}))),
            _tool("err2", ("return", _NonSerializable())),
            _tool("err3", ("raise", ValueError("boom"))),
        ],
        tmp_path,
    )
    calls = [_call("err1", "call-e1"), _call("err2", "call-e2"), _call("err3", "call-e3")]
    out = agent.execute_tools(_state(calls), _cfg("t-err"))

    entries = {e["tool_call_id"]: e for e in agent.ledger.get("t-err")}
    assert len(entries) == 3

    # call-e1: returned error tuple, marked failed via its own metadata
    assert entries["call-e1"]["raw_output"][1]["analysis_status"] == "failed"

    # call-e2: non-serializable return is string-coerced without raising
    assert isinstance(entries["call-e2"]["raw_output"], str)

    # call-e3: raise -> synthesized error tuple + error marker
    assert entries["call-e3"]["raw_output"] == [{"error": "boom"}, {"analysis_status": "failed"}]
    assert entries["call-e3"]["error"] is True

    # content on the raise path stays eval-parseable (interface.py:149 contract)
    msg_e3 = {m.tool_call_id: m for m in out["messages"]}["call-e3"]
    assert msg_e3.content == str(({"error": "boom"}, {"analysis_status": "failed"}))
    assert ast.literal_eval(msg_e3.content)[0] == {"error": "boom"}

    # run did not abort: all three messages emitted
    assert len(out["messages"]) == 3


# --- AC-7: log_tools=False disables both the log file and the ledger ----------
def test_ac7_log_tools_false_disables_ledger(tmp_path):
    agent = _agent(
        [_tool("chest_xray_classifier", _completed({"x": 0.1}))], tmp_path, log_tools=False
    )
    agent.execute_tools(_state([_call("chest_xray_classifier", "off-1")]), _cfg("t-off"))

    assert agent.ledger.get("t-off") == []
    assert not list(tmp_path.glob("tool_calls_*.json"))


# --- AC-8: ledger stored under log_dir, retention parity ----------------------
def test_ac8_ledger_under_log_dir(tmp_path):
    agent = _agent([_tool("chest_xray_classifier", _completed({"x": 0.1}))], tmp_path)
    agent.execute_tools(_state([_call("chest_xray_classifier", "r-1")]), _cfg("t-ret"))

    assert (tmp_path / "ledger.json").exists()
    entry = agent.ledger.get("t-ret")[0]
    assert entry["raw_output"][0] == {"x": 0.1}


class _ScriptedModel:
    """Emits one tool call on the first turn, then a plain answer to terminate."""

    def __init__(self, tool_name):
        self._tool_name = tool_name
        self._turn = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._turn += 1
        if self._turn == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": self._tool_name, "args": {"image_path": "a.png"}, "id": "g1", "type": "tool_call"}
                ],
            )
        return AIMessage(content="done")


# --- CD-2: thread_id flows through the compiled graph (langgraph config inject) -
def test_graph_injects_thread_id(tmp_path):
    agent = Agent(
        model=_ScriptedModel("chest_xray_classifier"),
        tools=[_tool("chest_xray_classifier", _completed({"x": 0.2}))],
        log_tools=True,
        log_dir=str(tmp_path),
    )
    agent.workflow.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t-graph"}},
    )
    entries = agent.ledger.get("t-graph")
    assert len(entries) == 1
    assert entries[0]["tool_call_id"] == "g1"
    assert entries[0]["provenance_source"] == "DenseNet-121"
