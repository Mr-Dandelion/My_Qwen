# app.py
from sglang import LLM, OpenAICompatibleApp

llm = LLM(model_path="/app/models/qwen3-8b", tensor_parallel_size=1)

app = OpenAICompatibleApp(llm)
