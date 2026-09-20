import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from rag.tools import tools

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    provider="together",
    huggingfacehub_api_token=HF_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
chat_with_tools = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "alfred-demo-thread-1"}}

SYSTEM_PROMPT = (
    "You are Alfred, a knowledgeable butler assisting with a gala. "
    "When asked about a guest, ALWAYS call guest_info_retriever first. "
    "Only call web_search if guest_info_retriever returns 'No matching "
    "guest information found.' When guest_info_retriever's output includes "
    "a conversation-starter instruction, include that conversation starter "
    "in your final answer to the user. "
    "Use get_weather_info when asked about current weather in a location. "
    "When asked about an organization or person's most downloaded or "
    "popular model, ALWAYS call get_hub_stats with that name as the author "
    "argument, even if you are not sure they publish models on the Hugging "
    "Face Hub. Never answer from your own knowledge or assume an author "
    "has no models without first calling get_hub_stats and checking its "
    "result."
)

turn1 = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'."),
]
response1 = alfred.invoke({"messages": turn1}, config=config)
print("🎩 Alfred's Response (guest lookup):")
print(response1["messages"][-1].content)

turn2 = [HumanMessage(content="What was the name of the guest I just asked about?")]
response2 = alfred.invoke({"messages": turn2}, config=config)
print("🎩 Alfred's Response (memory recall):")
print(response2["messages"][-1].content)

turn3 = [HumanMessage(content="Do we have a guest named 'Alan Turing'? Tell me about them.")]
response3 = alfred.invoke({"messages": turn3}, config=config)
print("🎩 Alfred's Response (web search fallback):")
print(response3["messages"][-1].content)

turn4 = [HumanMessage(content="Who is Facebook and what's their most downloaded and popular AI model?")]
response4 = alfred.invoke({"messages": turn4}, config=config)
print("🎩 Alfred's Response (hugging face hub fallback):")
print(response4["messages"][-1].content)

turn5 = [HumanMessage(content="What is the weather in Tel-Aviv now?")]
response5 = alfred.invoke({"messages": turn5}, config=config)
print("🎩 Alfred's Response (dummy weather fallback):")
print(response5["messages"][-1].content)