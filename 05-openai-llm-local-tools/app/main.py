import openai

cliente = openai.OpenAI(
    # base_url="http://qwen3-0.6b:8080/v1",
    base_url="http://qwen2.5-3b:8080/v1",
    api_key="local"
)

# TOOLS
tools = [
    {
        "type": "function",
        "function": {
            "name": "calcular_iva",
            "description": "Calcula el precio más IVA cuando se solicite",
            "parameters": {
                "type": "object",
                "properties": {
                    "precio": {
                        "type": "number",
                        "description": "El precio base"
                    },
                    "iva": {
                        "type": "number",
                        "description": "El IVA o 0.16 (16%)"
                    }
                },
                "required": [
                    "precio"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_inflacion",
            "description": (
                "Calcula la inflación entre el "
                "precio anterior y el precio actual"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precio_anterior": {
                        "type": "number",
                        "description": "El precio anterior"
                    },
                    "precio_actual": {
                        "type": "number",
                        "description": "El precio actual"
                    }
                },
                "required": [
                    "precio_anterior",
                    "precio_actual"
                ]
            }
        }
    },
]

# Petición de RESPUESTA GUIADA (tools)
respuesta = cliente.chat.completions.create(
    model="local",
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente que reconoce solicitudes de cálculos"
        },
        {
            "role": "user",
            "content": (
                "Hola, ayudas a determinar la inflación de la leche "
                "antes costaba $24.5 y ahora cuesta $42.5 "
                "¿Cuánto cuesta hoy con iva?"
            )
        },
    ],
    tools=tools,
    tool_choice="auto"
)

for choice in respuesta.choices:
    print(choice.message.content)
    print("-" * 80)
    if choice.message.tool_calls:
        for tool in choice.message.tool_calls:
            print(tool)
            print(tool.function.name)
            print(tool.function.arguments)
    else:
        print("No se reconoció ningún tool")
    print("=" * 80)