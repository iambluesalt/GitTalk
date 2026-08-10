"""
Tests for the LangGraph agent (agent_graph.py).

Two layers:
- `build_agent_run` setup logic (intent gate, history dedup, budgets) with the
  model/router/db mocked.
- The compiled agent<->tools graph itself, driven with a scripted fake model and
  real (in-memory) tools, to prove the multi-hop loop actually round-trips
  through ToolNode more than once before ending.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import tool

from models import ConversationTurn


# ═══════════════════════════════════════════════════════════════════════════════
# build_agent_run — setup / gating
# ═══════════════════════════════════════════════════════════════════════════════

def _project(name="testrepo", repo_map=None):
    project = MagicMock()
    project.name = name
    project.repo_map = repo_map
    return project


async def test_general_intent_skips_the_graph():
    """Small talk gets the bare chat model, no tools, no graph."""
    from services.agent_graph import build_agent_run

    bare_model = MagicMock(name="bare_chat_model")
    with patch("services.agent_graph.db") as mock_db, \
         patch("services.agent_graph.router_service") as mock_router, \
         patch("services.agent_graph.llm_service") as mock_llm:
        mock_db.get_project.return_value = _project()
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="general")
        mock_llm.get_chat_model.return_value = bare_model

        run = await build_agent_run("proj-1", "hi there")

    assert run.uses_tools is False
    assert run.intent == "general"
    assert run.runnable is bare_model
    # No tools bound for a general turn.
    assert mock_llm.get_chat_model.call_args.kwargs.get("tools") is None


async def test_code_intent_builds_a_tool_bound_graph():
    """A code question compiles the agent<->tools graph with a recursion budget."""
    from services.agent_graph import build_agent_run
    from config import settings

    model_with_tools = MagicMock(name="model_with_tools")
    with patch("services.agent_graph.db") as mock_db, \
         patch("services.agent_graph.router_service") as mock_router, \
         patch("services.agent_graph.llm_service") as mock_llm, \
         patch("services.agent_graph.make_tools") as mock_make_tools:
        mock_db.get_project.return_value = _project()
        mock_db.get_conversation_history.return_value = []
        mock_router.classify = AsyncMock(return_value="code")
        mock_llm.get_chat_model.return_value = model_with_tools
        mock_make_tools.return_value = []

        run = await build_agent_run("proj-1", "how does auth work?")

    assert run.uses_tools is True
    assert run.intent == "code"
    assert run.inputs["messages"][-1].content == "how does auth work?"
    assert run.config["recursion_limit"] == settings.AGENT_MAX_TOOL_HOPS * 2 + 1
    # Graph object, not a bare runnable — it exposes astream_events.
    assert hasattr(run.runnable, "astream_events")
    mock_make_tools.assert_called_once_with("proj-1")
    assert mock_llm.get_chat_model.call_args.kwargs.get("tools") == []


async def test_missing_project_falls_back_to_general():
    from services.agent_graph import build_agent_run

    bare_model = MagicMock(name="bare_chat_model")
    with patch("services.agent_graph.db") as mock_db, \
         patch("services.agent_graph.llm_service") as mock_llm:
        mock_db.get_project.return_value = None
        mock_llm.get_chat_model.return_value = bare_model

        run = await build_agent_run("missing-proj", "anything")

    assert run.intent == "general"
    assert run.uses_tools is False
    assert run.inputs == [HumanMessage(content="anything")]


async def test_trailing_duplicate_user_turn_is_dropped():
    """The route persists the user's message before calling build_agent_run, so
    it comes back as the last history turn; the agent path must not ask the same
    question twice."""
    from services.agent_graph import build_agent_run

    history = [
        ConversationTurn(role="user", content="earlier question"),
        ConversationTurn(role="assistant", content="earlier answer"),
        ConversationTurn(role="user", content="how does auth work?"),
    ]
    with patch("services.agent_graph.db") as mock_db, \
         patch("services.agent_graph.router_service") as mock_router, \
         patch("services.agent_graph.llm_service") as mock_llm, \
         patch("services.agent_graph.make_tools", return_value=[]):
        mock_db.get_project.return_value = _project()
        mock_router.classify = AsyncMock(return_value="code")
        mock_llm.get_chat_model.return_value = MagicMock()

        run = await build_agent_run(
            "proj-1", "how does auth work?", conversation_history=history,
        )

    contents = [m.content for m in run.inputs["messages"]]
    assert contents.count("how does auth work?") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# build_graph — the agent<->tools loop, multi-hop
# ═══════════════════════════════════════════════════════════════════════════════

class ScriptedModel:
    """Fake chat model: yields one scripted AIMessageChunk response per call,
    in order — enough to drive the real StateGraph through multiple hops."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def astream(self, messages):
        response = self._responses[self.calls]
        self.calls += 1
        yield response


def _tool_call_chunk(name, args, call_id):
    return AIMessageChunk(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


async def test_multi_hop_loop_runs_search_codebase_more_than_once():
    """A model that keeps asking to search must round-trip through ToolNode each
    time, and the final answer only appears once it stops calling tools."""
    from services.agent_graph import build_graph

    call_log = []

    @tool
    async def search_codebase(query: str) -> str:
        """Search the codebase."""
        call_log.append(("search_codebase", query))
        return f"results for {query}"

    @tool
    async def read_file(path: str) -> str:
        """Read a file."""
        call_log.append(("read_file", path))
        return f"contents of {path}"

    model = ScriptedModel([
        _tool_call_chunk("search_codebase", {"query": "auth flow"}, "1"),
        _tool_call_chunk("search_codebase", {"query": "login handler"}, "2"),
        _tool_call_chunk("read_file", {"path": "src/auth.py"}, "3"),
        AIMessageChunk(content="Auth works via a login handler in src/auth.py."),
    ])

    graph = build_graph(model, [search_codebase, read_file])
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="how does auth work?")]},
        config={"recursion_limit": 25},
    )

    search_calls = [c for c in call_log if c[0] == "search_codebase"]
    assert len(search_calls) >= 2
    assert ("read_file", "src/auth.py") in call_log

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "Auth works" in final.content
    # No leftover tool call on the answer that ended the loop.
    assert not final.tool_calls


async def test_graph_ends_immediately_when_no_tool_call_is_made():
    from services.agent_graph import build_graph

    @tool
    async def search_codebase(query: str) -> str:
        """Search the codebase."""
        return "unused"

    model = ScriptedModel([AIMessageChunk(content="No search needed, here's the answer.")])
    graph = build_graph(model, [search_codebase])

    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})

    assert model.calls == 1
    assert result["messages"][-1].content == "No search needed, here's the answer."
