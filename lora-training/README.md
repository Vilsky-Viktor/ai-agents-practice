# lora-training

LoRA fine-tuning of `google/gemma-2-2b-it` for function calling, using the [Hermes function-calling thinking dataset](https://huggingface.co/datasets/Jofthomas/hermes-function-calling-thinking-V1) and TRL/PEFT.

`train.py` loads the base model and tokenizer (customizing the chat template and adding ChatML-style special tokens), preprocesses the dataset into chat-formatted text, configures a `LoraConfig`, and fine-tunes via `SFTTrainer`.

## Running

```bash
poetry install
poetry run python train.py
```

Requires a GPU (bf16 model loading) and a Hugging Face token with access to `google/gemma-2-2b-it` — see `.env.example`. The fine-tuned adapter is saved to `gemma-2-2B-it-thinking-function_calling-V0/`.
