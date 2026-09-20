import re
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from smolagents import (CodeAgent, DuckDuckGoSearchTool, InferenceClientModel)
import uuid

load_dotenv()

langfuse = get_client()

if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

SmolagentsInstrumentor().instrument()

search_tool = DuckDuckGoSearchTool()

agent = CodeAgent(
    tools=[search_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
)

judge_model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

JUDGE_PROMPT = """You are evaluating the quality of an AI agent's response to a user's question.

Question: {question}
Agent's response: {response}

Rate the response's correctness and helpfulness on a scale from 1 (very poor) to 5 (excellent).
Respond in EXACTLY this format, nothing else:
SCORE: <number 1-5>
REASON: <one sentence explaining the score>"""


def judge_response(question: str, response: str) -> tuple[float, str]:
    prompt = JUDGE_PROMPT.format(question=question, response=response)
    output = judge_model([{"role": "user", "content": prompt}]).content

    score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", output)
    reason_match = re.search(r"REASON:\s*(.+)", output)
    score = float(score_match.group(1)) if score_match else 0.0
    reason = reason_match.group(1).strip() if reason_match else output.strip()
    return score, reason

session_id = f"session-{uuid.uuid4()}"
user_id = f"user-{uuid.uuid4()}"

print("\nChat with the agent. Type 'exit' or 'quit' to stop.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    with langfuse.start_as_current_observation(name="chat-turn", as_type="span") as span:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=["terminal-chat"],
        ):
            response = agent.run(question)

        span.update(input=question, output=response)

        print(f"\nAgent: {response}\n")

        with langfuse.start_as_current_observation(name="llm-judge", as_type="evaluator") as judge_span:
            judge_score, judge_reason = judge_response(question, response)
            judge_span.update(
                input={"question": question, "response": response},
                output=f"SCORE: {judge_score}\nREASON: {judge_reason}",
            )
        span.score_trace(
            name="llm-judge-quality",
            value=judge_score,
            data_type="NUMERIC",
            comment=judge_reason,
        )
        print(f"LLM Judge: {judge_score}/5 — {judge_reason}\n")

        feedback = input("Was this response helpful? [y/n/skip]: ").strip().lower()
        if feedback in ("y", "yes", "n", "no"):
            comment = input("Optional comment (press Enter to skip): ").strip() or None
            span.score_trace(
                name="user-feedback",
                value=1.0 if feedback in ("y", "yes") else 0.0,
                data_type="BOOLEAN",
                comment=comment,
            )
            print("Thanks for the feedback!\n")
        else:
            print()

    langfuse.flush()

print("Goodbye!")
