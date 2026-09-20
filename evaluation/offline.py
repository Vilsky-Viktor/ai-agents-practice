import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from langfuse import get_client
from smolagents import CodeAgent, InferenceClientModel

load_dotenv()

langfuse = get_client()

dataset = load_dataset("openai/gsm8k", 'main', split='train')
df = pd.DataFrame(dataset)
print("First few rows of GSM8K dataset:")
print(df.head())

dataset_name = "gsm8k_dataset_huggingface"
current_run_name = "notebook-run-01"

langfuse.create_dataset(
    name=dataset_name,
    description="GSM8K benchmark dataset uploaded from Huggingface",
    metadata={
        "date": "2025-03-10", 
        "type": "benchmark"
    }
)

for idx, row in df.iterrows():
    langfuse.create_dataset_item(
        dataset_name=dataset_name,
        input={"text": row["question"]},
        expected_output={"text": row["answer"]},
        metadata={"source_index": idx}
    )
    if idx >= 9: # Upload only the first 10 items for demonstration
        break

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True
)

def run_smolagent_task(*, item, **kwargs):
    return agent.run(item.input["text"])

dataset = langfuse.get_dataset(name=dataset_name)

result = dataset.run_experiment(
    name=current_run_name,
    run_name=current_run_name,
    description="Evaluation run for GSM8K dataset",
    metadata={"model_provider": "Hugging Face", "temperature_setting": 0.7},
    task=run_smolagent_task,
)

print(result.format())

