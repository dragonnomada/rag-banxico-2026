from pathlib import Path
import json

from pypdf import PdfReader

ruta_contratos = Path("shared/input/financiera/contratos")

ruta_textos = Path("shared/output/financiera/textos")
ruta_textos.mkdir(parents=True, exist_ok=True)

for carpeta in ruta_contratos.iterdir():
    if not carpeta.is_dir():
        continue

    folio = carpeta.name

    for contrato in carpeta.iterdir():
        if not contrato.is_file():
            continue
        if contrato.name.startswith("._"):
            continue
        if not contrato.name.endswith(".pdf"):
            continue

        print(contrato)

        pdf = PdfReader(contrato)

        paginas = []

        for numero, pagina in enumerate(pdf.pages, start=1):
            texto = pagina.extract_text() or "SIN TEXTO"
            paginas.append({
                "folio": folio,
                "pagina": numero,
                "texto": texto
            })

        ruta_textos_contratos = Path(ruta_textos / "contratos")
        ruta_textos_contratos.mkdir(parents=True, exist_ok=True)

        ruta_textos_contratos_contrato = Path(
            ruta_textos_contratos / f"{folio}.json"
        )

        ruta_textos_contratos_contrato.write_text(
            json.dumps({
                "folio": folio,
                "ruta": f"{contrato}",
                "paginas": paginas
            }, indent=2),
            encoding="utf-8"
        )
        

        