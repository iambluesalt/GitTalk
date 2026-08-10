"""
LangGraph tool-calling agent.

Replaces the retrieve-once-then-answer pipeline in `rag_service.py`: instead of
injecting search results into the user message before the first model call, the
model asks for code itself via tools and can search/read repeatedly before
answering.

`build_agent_run()` does the per-request setup (intent gate, system prompt,
history windowing) and hands the route a runnable it can drive with
`astream_events`. The route owns SSE translation.
"""
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import settings
from logger import logger
from models import ConversationTurn
from services.agent_tools import make_tools
from services.llm_service import get_prompt_template, llm_service
from services.router_service import router_service
from storage.metadata_db import db


# Fixed overheads reserved outside the repo-map / history budgets.
_SYSTEM_FIXED_TOKENS = 500
_QUERY_FIXED_TOKENS = 200


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    return len(text) // 4


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


# ============================================================================
# Context assembly
# ============================================================================

def _budgets(intent: str) -> tuple[int, int]:
    """Return (repo_map_budget, history_budget) in tokens for an intent.

    Unlike the legacy pipeline there is no up-front chunk budget — retrieved code
    arrives as tool results mid-run. Whatever isn't spent here is what the tool
    results get to use, so both budgets stay deliberately modest for code
    queries.
    """
    remaining = settings.MAX_CONTEXT_TOKENS - _SYSTEM_FIXED_TOKENS - _QUERY_FIXED_TOKENS

    if intent == "general":
        # No tools will run; spend the context on conversation instead.
        return int(remaining * 0.15), int(remaining * 0.75)
    if intent == "follow_up":
        return int(remaining * 0.10), int(remaining * 0.40)
    # "code" — leave the majority of the window for tool results.
    return int(remaining * 0.18), int(remaining * 0.25)


def build_system_message(
    project_name: str,
    repo_map: str | None,
    repo_map_budget: int,
    intent: str,
) -> str:
    """Build the system prompt, adapting to query intent."""
    if intent == "general":
        parts = [get_prompt_template("general", project_name)]
    else:
        parts = [get_prompt_template("code_agent", project_name)]
        if intent == "follow_up":
            parts.append(
                "\nThe user is following up on a previous topic. Use the conversation "
                "history to resolve what they're referring to before searching."
            )

    if repo_map:
        parts.append("")
        parts.append("## Repository Structure")
        parts.append(_truncate_to_tokens(repo_map, repo_map_budget))

    return "\n".join(parts)


def build_history_messages(
    history: list[ConversationTurn],
    budget: int,
) -> list[BaseMessage]:
    """Build history messages within a token budget, keeping the newest turns."""
    if not history:
        return []

    selected: list[BaseMessage] = []
    tokens_used = 0

    # Walk backwards (newest first) so the most recent context survives.
    for turn in reversed(history):
        turn_tokens = _estimate_tokens(turn.content)
        if tokens_used + turn_tokens > budget:
            break
        message = (
            AIMessage(content=turn.content)
            if turn.role == "assistant"
            else HumanMessage(content=turn.content)
        )
        selected.append(message)
        tokens_used += turn_tokens

    selected.reverse()
    return selected


# ============================================================================
# Graph
# ============================================================================

def build_graph(model_with_tools: Runnable, tools: list) -> Any:
    """Compile the agent ⇄ tools loop.

    The agent node streams rather than invokes so `astream_events` emits
    `on_chat_model_stream` for the final answer; chunks are summed back into a
    single AIMessage (chunk addition merges partial tool-call arguments).
    """

    async def agent_node(state: MessagesState) -> dict:
        accumulated = None
        async for chunk in model_with_tools.astream(state["messages"]):
            accumulated = chunk if accumulated is None else accumulated + chunk
        if accumulated is None:
            accumulated = AIMessage(content="")
        return {"messages": [accumulated]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" when the model asked for one, else END.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ============================================================================
# Per-request assembly
# ============================================================================

@dataclass
class AgentRun:
    """Everything the route needs to stream one turn."""

    runnable: Runnable
    inputs: Any
    intent: str
    uses_tools: bool
    config: dict = field(default_factory=dict)
    token_count: int = 0


async def build_agent_run(
    project_id: str,
    query: str,
    conversation_id: str | None = None,
    conversation_history: list[ConversationTurn] | None = None,
    model_override: str | None = None,
) -> AgentRun:
    """Assemble the runnable and initial messages for one user turn.

    Small talk is gated out before the graph is built: a "general" intent skips
    tools entirely and streams straight from the chat model, which is exactly
    what the legacy pipeline did.
    """
    if conversation_id and conversation_history is None:
        conversation_history = db.get_conversation_history(conversation_id)
    conversation_history = conversation_history or []

    # The route persists the user's message before calling us, so it comes back
    # as the last history turn. Legacy left the duplicate in place; a tool-calling
    # loop reads it as the question being asked twice, so drop it.
    if (
        conversation_history
        and conversation_history[-1].role == "user"
        and conversation_history[-1].content == query
    ):
        conversation_history = conversation_history[:-1]

    project = db.get_project(project_id)
    if not project:
        logger.error(f"Project not found: {project_id}")
        return AgentRun(
            runnable=llm_service.get_chat_model(model_override),
            inputs=[HumanMessage(content=query)],
            intent="general",
            uses_tools=False,
            token_count=_estimate_tokens(query),
        )

    recent_for_router = [
        {"role": t.role, "content": t.content} for t in conversation_history[-4:]
    ]
    intent = await router_service.classify(query, recent_for_router)
    logger.info(f"Query intent: {intent} | query: {query[:80]}")

    repo_map_budget, history_budget = _budgets(intent)
    messages: list[BaseMessage] = [
        SystemMessage(
            content=build_system_message(
                project.name, project.repo_map, repo_map_budget, intent
            )
        )
    ]
    messages.extend(build_history_messages(conversation_history, history_budget))
    messages.append(HumanMessage(content=query))

    token_count = sum(_estimate_tokens(str(m.content)) for m in messages)

    if intent == "general":
        return AgentRun(
            runnable=llm_service.get_chat_model(model_override),
            inputs=messages,
            intent=intent,
            uses_tools=False,
            token_count=token_count,
        )

    tools = make_tools(project_id)
    model_with_tools = llm_service.get_chat_model(model_override, tools=tools)
    return AgentRun(
        runnable=build_graph(model_with_tools, tools),
        inputs={"messages": messages},
        intent=intent,
        uses_tools=True,
        # Each hop is two graph steps (agent + tools); +1 for the final answer.
        config={"recursion_limit": settings.AGENT_MAX_TOOL_HOPS * 2 + 1},
        token_count=token_count,
    )
