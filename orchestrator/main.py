""" **What we are doing**: LangGraph as its own container/service
```
/run endpoint → HTTP call → orchestrator service → MCP client → MCP server
```
More realistic for production. Teaches you more Docker. But adds complexity now.
"""

""" **What the LangGraph agent will do**:
For Phase 4, the agent will be a ReAct-style agent (Reason + Act loop). This is the standard pattern:
```
user message
    ↓
LLM thinks: "I should call profile_dataset first"
    ↓
calls profile_dataset via MCP
    ↓
LLM thinks: "Now I should detect the problem type"
    ↓
calls detect_problem_type
    ↓
... continues until it decides it's done
    ↓
returns final answer
```
LangGraph has a prebuilt `create_react_agent` for exactly this. We'll start there (not a custom graph) because it gets you observability in LangSmith immediately, and you can see the reasoning trace.
"""

""" What we are adding:
What we're adding
One new service: orchestrator. Here's where it slots into your existing compose:
```
curl POST /run  →  backend:8000  →  orchestrator:8002  →  mcp-server:8001
                                          ↕
                                    Gemini 3.1 Flash Lite
                                          ↕
                                    LangSmith (traces out)
```

The backend's /run endpoint stops calling MCP directly. Instead it makes an HTTP POST to the orchestrator. The orchestrator runs the LangGraph ReAct agent, which calls MCP tools as needed.
"""

"""
`main.py` will be a FastAPI app with one endpoint: `POST /invoke`. It takes `{"message": "..."}` and returns the agent's final response.
The agent uses three `LangGraph`/`LangChain` primitives. Here's what each does:
* `MultiServerMCPClient` — from `langchain-mcp-adapters`. You give it your MCP server URL and transport type. It connects, does the MCP handshake, and returns your tools as standard `LangChain` tool objects. This replaces all the raw `streamablehttp_client` code.
* `create_react_agent` — prebuilt `LangGraph` function. Takes an LLM + a list of tools, returns a compiled graph that implements the ReAct loop. The loop is: LLM thinks → decides to call a tool → tool executes → result goes back to LLM → repeat until LLM says it's done.
* `MessagesState` — the state that flows through the graph. It's just a list of messages `(HumanMessage, AIMessage, ToolMessage)` that grows as the loop runs. `LangSmith` shows you this entire conversation trace.
"""

"""
Why the orchestrator is fully `async`:
In Phase 3, your FastAPI backend was mostly synchronous — request comes in, call MCP, return result. That worked because you were making one sequential call.
The LangGraph agent is different. `MultiServerMCPClient` and `create_react_agent` are async-native — they use Python's async/await system. This means:
* The MCP client connection uses `async with`
* Tool loading uses await `client.get_tools()`
* Agent invocation uses `await agent.ainvoke(...)`

FastAPI handles this cleanly — if you declare an endpoint as `async def`, FastAPI runs it in the async event loop rather than a thread pool. So the orchestrator's `main.py` will look slightly different from what you wrote before, but the pattern is consistent throughout.
> When you see `async def` and `await`, read it as: "this operation might take time (network call, LLM response), so Python can do other things while waiting instead of blocking."
"""

from curses import raw
import os
import asyncio
from contextlib import asynccontextmanager          # lets us define startup/shutdown logic for the FastAPI app.

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient          # the adapter that connects to your MCP server and converts its tools into LangChain tools.
from langgraph.prebuilt import create_react_agent               # prebuilt LangGraph function that builds the ReAct loop graph from an LLM + tools.

"""
What's happening here, concept by concept:
* `lifespan`is a FastAPI pattern for startup/shutdown logic. Everything before yield runs when the container starts. Everything after yield runs when it shuts down. The `app = FastAPI(lifespan=lifespan)` line wires it in.
* `global agent` — we're storing the compiled agent in a module-level variable so every request can use it without rebuilding. This is the right pattern for expensive-to-construct objects.
* `MultiServerMCPClient` — you give it a dict of named servers. "automl" is just a label. The URL and transport tell it how to reach your MCP server. Note the `/mcp` path suffix — same gotcha as Phase 3.
* `await client.get_tools()` — this does the MCP handshake and returns your four stub tools as LangChain tool objects. After this line, tools is a list the LLM can reason about.
* `create_react_agent(llm, tools)` — compiles the ReAct graph. The result is a runnable agent. No loop logic for you to write — LangGraph handles the `think→call→observe` cycle internally.
"""
# Global agent — initialized once at startup
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent

    mcp_base_url = os.environ["MCP_SERVER_URL"]

    client = MultiServerMCPClient({
        "automl": {
            "url": f"{mcp_base_url}/mcp",
            "transport": "streamable_http",
        }
    })

    # Retry loop — mcp-server may not be ready immediately at startup
    max_retries = 10
    retry_delay = 3  # seconds between attempts

    for attempt in range(1, max_retries + 1):
        try:
            tools = await client.get_tools()
            print(f"MCP tools loaded successfully on attempt {attempt}.")
            break
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries}: MCP server not ready yet — {e}")
            if attempt == max_retries:
                raise RuntimeError("MCP server unreachable after max retries. Exiting.") from e
            await asyncio.sleep(retry_delay)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    agent = create_react_agent(llm, tools)

    yield  # app runs here

    # shutdown: nothing to explicitly close for streamable_http


app = FastAPI(lifespan=lifespan)

"""
What's happening:
* `InvokeRequest`— Pydantic model, same pattern as Phase 1. Validates that the request body has a message field.
* `agent.ainvoke(...)` — this is the async version of invoking the graph. You pass in the initial state: a list with one user message. The agent runs its full ReAct loop internally and returns the final state — which is the same list of messages, now extended with all the AI reasoning steps and tool calls that happened.
* `result["messages"][-1].content` — we grab only the last message, which is the LLM's final response after all tool calls are done. Everything in between (tool calls, tool results, intermediate reasoning) is visible in LangSmith but we don't need to return it all to the caller.
* One thing to be aware of: If the agent errors mid-loop (e.g. Gemini returns an empty response, which is the known 2.5-series bug), this line will raise an exception. FastAPI will return a 500. That's actually fine for now — it'll give us a clear signal to debug.
"""
class InvokeRequest(BaseModel):
    message: str


@app.post("/invoke")
async def invoke(request: InvokeRequest):
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": request.message}]
    })

    # The agent returns a list of messages — the last one is the final response
    raw = result["messages"][-1].content
    if isinstance(raw, list):
        final_message = " ".join(
            block["text"] for block in raw if block.get("type") == "text"
        )
    else:
        final_message = raw

    return {"response": final_message}