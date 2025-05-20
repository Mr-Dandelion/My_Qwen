from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.76.222:8000/v1",
    api_key="sk-local"      # 任意非空字符串
)

resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "用三句话夸夸我"}],
    temperature=0.6,
)
print(resp.choices[0].message.content)
