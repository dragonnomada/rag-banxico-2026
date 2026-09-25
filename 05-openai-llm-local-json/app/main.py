import openai

cliente = openai.OpenAI(
    base_url="http://qwen3-0.6b:8080/v1",
    api_key="local"
)

# Petición de RESPUESTA CERRADA (esquema json)
respuesta = cliente.chat.completions.create(
    model="local",
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente de un restaurante de comida"
        },
        {
            "role": "user",
            "content": (
                "Hola, quiero una pizza grande de peperoni "
                "con queso extra, y chiles jalapeños "
                "y también una coca de 600 bien fria"
            )
        },
    ],
    response_format={
        "type": "json_schema",
        "schema": {
            "type": "array",
            "properties": {
                "producto": {
                    "type": "string",
                    "description": "Reconoce la comida o bebida"
                },
                "tipo": {
                    "type": "string",
                    "description": "Tipo de producto, por ejemplo, tipo de pizza o sabor de la bebida"
                },
                "extras": {
                    "type": "string",
                    "description": "Adicionales del producto"
                },
                "tamaño": {
                    "type": "string",
                    "descripcion": "Tamaño del producto, si es grande, mediana, chica"
                },
                "cantidad": {
                    "type": "number",
                    "descripcion": "Cantidad en mililitros o litros si aplica"
                },
            },
            "required": [
                "producto",
                "tipo"
            ]
        }
    }
)

for choice in respuesta.choices:
    print(choice.message.content)
    print("-" * 80)