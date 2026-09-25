import openai

cliente = openai.OpenAI(
    base_url="http://qwen3-0.6b:8080/v1",
    api_key="local"
)

# Petición de RESPUESTA ABIERTA (chat)
respuesta = cliente.chat.completions.create(
    model="local",
    messages=[
        {
            "role": "system",
            "content": "Eres un filósofo especializado en teología"
        },
        {
            "role": "user",
            "content": "¿Existe Dios?"
        },
    ]
)

for choice in respuesta.choices:
    print(choice.message.content)
    print("-" * 80)