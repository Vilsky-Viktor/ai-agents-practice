import random
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from rag.retriever import hybrid_retriever
from huggingface_hub import list_models

TOP_K = 3


def guest_info_lookup(query: str) -> str:
    """Retrieve guest info via hybrid (BM25 + embedding) search, plus a
    conversation-starter instruction for the calling LLM."""
    results = hybrid_retriever.invoke(query)
    if not results:
        return "No matching guest information found in the local guest list."

    info = "\n\n".join(doc.page_content for doc in results[:TOP_K])
    instruction = (
        "\n\nInstruction to assistant: Using the Description field above, "
        "craft one short, personalized conversation-starter you could use "
        "with this guest, referencing their specific interests or work. "
        "Include it in your final answer to the user."
    )
    return info + instruction

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads (descending by default)
        models = list(list_models(author=author, sort="downloads", limit=1))

        # Hugging Face Hub author/org IDs are case-sensitive (e.g. "facebook", not "Facebook")
        if not models and author != author.lower():
            models = list(list_models(author=author.lower(), sort="downloads", limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=guest_info_lookup,
    description=(
        "Retrieves detailed information (name, relation, description, email) "
        "about gala guests from the local guest list, using combined keyword "
        "and semantic search, and returns a suggested personalized "
        "conversation starter. ALWAYS try this tool FIRST for any question "
        "about a specific guest. Only use web_search if this tool returns "
        "'No matching guest information found.'"
    ),
)

web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Searches the web via DuckDuckGo for information about people NOT "
        "found in the local guest list. Use this ONLY after "
        "guest_info_retriever has returned 'No matching guest information "
        "found.'"
    ),
)

tools = [guest_info_tool, web_search_tool, weather_info_tool, hub_stats_tool]
