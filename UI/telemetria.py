from __future__ import annotations
import csv, os, time
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, Any, List

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class TelemetryLogger:
    """
    Registra cada consulta en un CSV (append).
    Escribe cabecera si el archivo no existe o está vacío.
    """
    DEFAULT_FIELDS = [
        "ts_start", "ts_end", "lat_ms",
        "query", "auto", "estado", "municipio",
        "filtros", "hits_count", "empty_result", "ok",
        "kb_version", "model_name"
    ]

    def __init__(self, csv_path: str, fieldnames: List[str] | None = None):
        self.csv_path = csv_path
        self.fieldnames = fieldnames or self.DEFAULT_FIELDS
        _ensure_dir(csv_path)

    def append(self, row: Dict[str, Any]):
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                w.writeheader()
            # filtra llaves desconocidas
            filtered = {k: row.get(k, "") for k in self.fieldnames}
            w.writerow(filtered)

@dataclass
class SessionMetrics:
    """
    Acumula métricas de la sesión (RAM) para la pestaña Supervisión.
    """
    total_queries: int = 0
    success_count: int = 0
    empty_count: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    top_queries: Counter = field(default_factory=Counter)

    def record(self, query: str, ok: bool, empty_result: bool, lat_ms: float):
        self.total_queries += 1
        if ok: self.success_count += 1
        if empty_result: self.empty_count += 1
        self.latencies_ms.append(float(lat_ms))
        if query:
            self.top_queries[query.strip().lower()] += 1

    def snapshot(self) -> Dict[str, Any]:
        import numpy as np
        lat = self.latencies_ms or [0.0]
        arr = np.array(lat, dtype=float)
        total = self.total_queries or 0
        success = self.success_count or 0
        return {
            "total": total,
            "success": success,
            "empty": self.empty_count,
            "success_rate": (success/total*100.0) if total else 0.0,
            "lat_p50": float(np.percentile(arr, 50)),
            "lat_p95": float(np.percentile(arr, 95)),
            "avg_lat": float(arr.mean()),
            "top_queries": self.top_queries.most_common(10)
        }
