# Modelos Locales - LLM

## Lista comparativa

- **8 GB de RAM:** Qwen 3 1.7B o Qwen 2.5 3B.
- **16 GB de RAM:** Gemma 3 4B, Phi 3.5 Mini o Llama 3.2 3B.
- **Mejor español:** Qwen 2.5 3B o Gemma 3 4B.
- **Mayor velocidad:** Qwen 3 0.6B o SmolLM2 1.7B.
- **Código y razonamiento:** Phi 3.5 Mini.

## Cuantizaciones

| Formato | Descripción | Calidad | Tamaño |
|---|---|---:|---:|
| `Q4_K_S` | Cuantización K de 4 bits, pequeña | Buena | Menor |
| `Q4_K_M` | Cuantización K de 4 bits, equilibrada | Muy buena | Medio |
| `Q4_K_L` | Cuantización K de 4 bits, grande | Algo mejor | Mayor |
| `Q4_0` | Cuantización antigua de 4 bits | Aceptable | Pequeño |
| `F16` | Pesos en coma flotante de 16 bits | Casi original | Muy grande |
| `F32` | Pesos en coma flotante de 32 bits | Original | Enorme |

## Qwen 3 - 0.6B / Q4

Modelo ligero, para pruebas iniciales.

> Peso aproximado: `409MB`

```bash
curl -L \
  -o Qwen3-0.6B-Q4_0.gguf \
  https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf
```

> Versiones: https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/tree/main

## Qwen 3 — 0.6B / Q8

Muy rápido y ligero, útil para tareas sencillas.

> Peso aproximado: `805MB`

```bash
curl -L \
  -o Qwen3-0.6B-Q8_0.gguf \
  "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
```

> Versiones: https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/tree/main

## Qwen 3 — 1.7B / Q4_K_M

Mejor equilibrio dentro de la familia Qwen 3. 

> Peso aproximado: `1.28 GB`

```bash
curl -L \
  -o Qwen3-1.7B-Q4_K_M.gguf \
  "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"
```

> Versiones: https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/tree/main

## Qwen 2.5 — 3B Instruct / Q4_K_M

Buena opción multilingüe y para programación básica.

> Peso aproximado: `1.93 GB`

```bash
curl -L \
  -o Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
```

> Versiones: https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/tree/main

## Llama 3.2 — 3B Instruct / Q4_K_M

Buen modelo general y con soporte razonable para español.

> Peso aproximado: `2.02 GB`

```bash
curl -L \
  -o Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

## Gemma 3 — 4B Instruct / Q4_K_M

De los más capaces, se puede subir un archivo adicional de visión.

> Peso aproximado: `2.49 GB`

```bash
curl -L \
  -o gemma-3-4b-it-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf"
```

## Phi 3.5 Mini Instruct / Q4_K_M

Especialmente bueno para razonamiento y programación en relación con su tamaño.

> Peso aproximado: `2.39 GB`

```bash
curl -L \
  -o Phi-3.5-mini-instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
```

## SmolLM2 — 1.7B Instruct / Q4_K_M

Muy ligero y rápido, aunque principalmente orientado al inglés. 

> Peso aproximado: `1.06 GB`

```bash
curl -L \
  -o SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
```