# ui_qt/controller.py
import csv
import time
from PySide6.QtCore import QThreadPool, QTimer
from UI.workers import AskWorker
from UI.formatters import format_answer, format_error
from UI.telemetria import TelemetryLogger, SessionMetrics

class AppController:
    def __init__(self, view, retrieval_module, kb_df, llm_model=None,
                 on_query_start=None, on_query_end=None,
                 telemetry_logger: TelemetryLogger | None = None,
                 session_metrics: SessionMetrics | None = None,
                 kb_version: str = "", model_name: str = ""):
        self.view = view
        self.llm_model = llm_model
        self.retrieval = retrieval_module
        self.kb_df = kb_df
        self.on_query_start = on_query_start
        self.on_query_end = on_query_end
        self.tlog = telemetry_logger
        self.metrics = session_metrics or SessionMetrics()
        self.kb_version = kb_version
        self.model_name = model_name
        self.pool = QThreadPool.globalInstance()
        self._last_hits_rows = []
        self.pool = QThreadPool.globalInstance()
        self._job_id = 0
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)  # ms // Usable cuando querramos implementar que busque mientras typeas
        self._debounce.timeout.connect(self._do_search)

        # conexiones UI
        self.view.combo_estado.currentTextChanged.connect(self._handle_estado_change)
        self.view.btn_limpiar.clicked.connect(self._handle_limpiar)
        self.view.btn_salir.clicked.connect(self._handle_salir)
        self.view.btn_buscar.clicked.connect(self._handle_buscar_clicked)
        self.view.input_query.returnPressed.connect(self._handle_buscar_clicked)
        self.view.btn_export.clicked.connect(self._handle_export)
        self.view.btn_export.clicked.connect(self._handle_export)
        self.view.btn_export_last.clicked.connect(self._handle_export_last_hits)
        self.view.btn_buscar.clicked.connect(self._handle_buscar_clicked)
        self.view.btn_cancel.clicked.connect(self._handle_cancel)

        # inicializa panel sup
        self.view.update_metrics_view(self.metrics.snapshot())

        #self.view.input_query.textChanged.connect(self._maybe_debounce)  ##  Implementacion de busqueda mientras typeas

    # Eventos UI
    def _handle_export_last_hits(self):
        """Exporta a CSV los hits de la última búsqueda."""
        if not self._last_hits_rows:
            self.view.append_result("⚠️ No hay resultados recientes para exportar.")
            return
        path = self.view.ask_export_last_hits_path()
        if not path:
            return
        # columnas en orden
        cols = [
            "id", "tipo_sede", "programa", "estado", "municipio",
            "direccion_corta", "horarios_texto", "telefono",
            "servicios", "score"
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in self._last_hits_rows:
                    w.writerow({k: row.get(k, "") for k in cols})
            self.view.append_result(f"✔ Últimos resultados exportados a: {path}")
        except Exception as e:
            self.view.append_result(f"[ERROR] No se pudo exportar últimos resultados: {e}")

    def _handle_export(self):
        path = self.view.ask_export_path()
        if not path or not self.tlog:
            return  # sin logger, nada que exportar
        # Aquí no reescribimos el CSV existente; solo informamos al usuario
        # que ya hay archivo en self.tlog.csv_path, o podrías copiarlo a 'path'.
        import shutil, os
        try:
            shutil.copyfile(self.tlog.csv_path, path)
            self.view.append_result(f"✔ Telemetría exportada a: {path}")
        except Exception as e:
            self.view.append_result(f"[ERROR] No se pudo exportar: {e}")

    def _handle_estado_change(self, estado_text: str):
        if not estado_text:
            self.view.update_municipios([])
            return
        sub = self.kb_df[self.kb_df["estado"].astype(str).str.lower() == estado_text.lower()]
        munis = sorted(set(str(x) for x in sub["municipio"].dropna().unique()))
        self.view.update_municipios(munis)

    def _handle_limpiar(self):
        self.view.clear_all()

    def _handle_salir(self):
        self.view.close()

    def _handle_buscar_clicked(self):
        self._do_search()

    def _maybe_debounce(self):
        if getattr(self.view, "check_live", None) and self.view.check_live.isChecked():
            self._debounce.start()

    def _handle_cancel(self):
        # invalidar el trabajo en curso (job_id sube)
        self._job_id += 1
        self.view.set_busy(False)
        self.view.append_result("⛔ Búsqueda cancelada por el usuario.\n")

    def _do_search(self):
        query = self.view.input_query.text().strip()
        auto = self.view.check_auto_location.isChecked()
        estado = self.view.combo_estado.currentText().strip()
        municipio = self.view.combo_muni.currentText().strip()
        fastpath = self.view.is_fastpath_enabled()
        if fastpath: self.retrieval.fast_path = fastpath

        if not query and not estado and not municipio:
            self.view.append_result("⚠️ Escribe una consulta o selecciona estado/municipio.")
            return

        filtros = {}
        if not auto:
            if estado: filtros["estado"] = estado
            if municipio: filtros["municipio"] = municipio

        # marca ocupado y genera un job_id
        self.view.set_busy(True)
        self._job_id += 1
        job_id = self._job_id

        t0 = time.time()
        if self.on_query_start:
            self.on_query_start({"query": query, "auto": auto, "filtros": filtros.copy(), "ts": t0})

        worker = AskWorker(self.retrieval, self.llm_model, query, filtros, job_id)

        def on_ok(resp_dict):
            # si la respuesta no corresponde al job actual, la ignoramos
            if resp_dict.get("_job_id") != self._job_id:
                return
            # ya no estamos ocupados
            self.view.set_busy(False)

            text = format_answer(resp_dict)
            self.view.append_result(text)

        def on_err(tb_text):
            # si el error no corresponde al job actual, ignorar
            if job_id != self._job_id:
                return
            self.view.set_busy(False)
            self.view.append_result(format_error(Exception(tb_text)))

        worker.signals.finished.connect(on_ok)
        worker.signals.error.connect(on_err)
        self.pool.start(worker)
