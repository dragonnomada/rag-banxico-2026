from pathlib import Path
import json

ruta_contratos_textos = Path("../../shared/output/financiera/textos/contratos")
ruta_contratos_chunks_carpeta = Path("../../shared/output/financiera/chunks")
ruta_contratos_chunks_contratos = Path(ruta_contratos_chunks_carpeta / "contratos.json")

ruta_contratos_chunks_carpeta.mkdir(parents=True, exist_ok=True)

contador = 1

chunks = []

for contrato in ruta_contratos_textos.iterdir():
    if not contrato.is_file():
        continue

    print(contrato)

    contrato_json = json.loads(contrato.read_text()) # { folio, ruta, paginas }

    # print(contrato_json)

    folio = contrato_json["folio"]

    for pagina in contrato_json["paginas"]: # { folio, pagina, texto }
        chunks.append({
            "folio": folio,
            "ruta": contrato_json["ruta"],
            "pagina": pagina["pagina"],
            "linea": 0,
            "chunk": contador,
            "texto": f"FOLIO {folio} PÁGINA {pagina['pagina']}"
        })
        contador += 1

        lineas = list(pagina["texto"].split("\n"))

        for linea, texto in enumerate(lineas): # línea
            chunks.append({
                "folio": contrato_json["folio"],
                "ruta": contrato_json["ruta"],
                "pagina": pagina["pagina"],
                "linea": linea + 1,
                "chunk": contador,
                "texto": f"FOLIO {folio} " + texto + " " + " ".join(lineas[linea+1:linea+5])
            })
            contador += 1

ruta_contratos_chunks_contratos.write_text(json.dumps(chunks, indent=2))
