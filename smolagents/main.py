from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool
import datetime
import time

login()

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
)

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

agent2 = CodeAgent(
    tools=[suggest_menu], 
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
)

agent2.run("Prepare a formal menu for the party.")

agen3 = CodeAgent(
    tools=[], 
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"), 
    additional_authorized_imports=['datetime']
)

agen3.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)