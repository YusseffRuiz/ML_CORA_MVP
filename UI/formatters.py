from typing import Dict

def format_answer(resp: Dict) -> str:
    """
    resp: {"question": str, "answer": str, "hits": [{"metadata":..., "score":...}, ...]}
    """
    if not isinstance(resp, dict):
        return str(resp)

    q = resp.get("question", "")
    ans = resp.get("answer", "")
    lines = [f"Consulta: {q}", "", ans]
    return "\n".join(lines)

def format_error(e: Exception) -> str:
    return f"[ERROR]\n{type(e).__name__}: {e}"


# Para revisar si se guardo correctamente
def format_doc_line(meta: dict) -> str:
    tel_display = meta.get("telefono") or ""
    servicios = ", ".join(meta.get("servicios_lista", []) or []) or "Consultar en sede"
    comidas = meta.get("comidas") or {}
    comidas_str = ""
    if isinstance(comidas, dict) and comidas:
        piezas = [f"{k}: {v}" for k,v in comidas.items()]
        comidas_str = " | Comidas: " + "; ".join(piezas)

    ume = meta.get("servicios_especiales") or []
    ume_str = f" | UME: {', '.join(ume[:6])}" if ume else ""

    movil_str = " | Unidad móvil" if meta.get("movil") else ""

    # line = (
    #     f"• {meta.get('id')} — {meta.get('municipio')}, {meta.get('estado')} | {meta.get('direccion_corta')} | "
    #     f"Horario: {meta['horarios_texto']} | Tel: {meta['telefono']} | "
    #     f"Horario de Comidas: {meta["comidas"]} | "
    #     f"Servicios: {', '.join(meta.get('servicios_lista', [])) or 'Consultar en sede'} | "
    #     f"Consulta: {meta.get('costo_consulta') or 'Consultar en sede'} | "
    #     f"Medicamentos: {meta.get('costo_medicamentos') or 'Consultar en sede'} "
    #     f"Datos sobre los médicos extra: {meta.get('datos_extras_medicos') or ' '} "
    #
    #
    # )

    line = (
        f"• {meta['id']} — {meta['municipio']}, {meta['estado']} | {meta['direccion_corta']} | "
        # f"Horario: {meta['horarios_texto']}"#| Tel: {meta['telefono']} | "
        # f"Horario de Comidas: {m['comidas']} | "
        f"Servicios: {', '.join(meta.get('servicios_lista', [])) or 'Consultar en sede'} | "
        # f"Consulta: {m.get('costo_consulta') or 'Consultar en sede'} | "
        # f"Medicamentos: {m.get('costo_medicamentos') or 'Consultar en sede'} "
        # f"Datos sobre los médicos extra: {m.get('datos_extras_medicos')}"
    )
    return line