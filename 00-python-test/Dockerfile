# Imagen base (ubuntu, debian, fedora, ..., python, node, jdk, ...)
FROM python:3.12.4-slim

# Directorio de trabajo
WORKDIR /app

# Copia los archivos locales al contenedor (imagen)
COPY app/* .

# Comando de ejecución al iniciar el contenedor
CMD ["python", "main.py"]
