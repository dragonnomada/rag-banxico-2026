with open("/shared/input/hola.txt", "r") as archivo_entrada:
    texto = archivo_entrada.read()

    with open("/shared/output/saludo.txt", "w") as archivo_salida:
        archivo_salida.writelines([
            "Hola desde python",
            "\n",
            "Tu archivo contiene:",
            "\n",
            "-" * 80,
            "\n",
            texto,
            "\n",
            "-" * 80,
            "\n",
            "Adiós 👌"
            "\n",
        ])