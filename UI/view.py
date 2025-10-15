# -*- coding: utf-8 -*-
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit, QComboBox, QCheckBox,
    QPushButton, QGridLayout, QHBoxLayout, QVBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog
)
from PySide6.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self, estados: list[str]):
        super().__init__()
        self.setWindowTitle("CORA - Centro de Orientación y Respuesta Automatizada")
        self.resize(1000, 700)

        # ---- Tab: Búsqueda ----
        title = QLabel("CORA : Asistente CAT (Retrieval) — MVP")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))

        self.input_query = QLineEdit()
        self.input_query.setPlaceholderText("Escribe tu consulta (ej. 'Farmacia en Puebla'...)")
        self.check_auto_location = QCheckBox("Detectar estado/municipio desde la consulta")
        self.check_auto_location.setChecked(True)
        self.check_fastpath = QCheckBox("⚡")
        self.check_fastpath.setChecked(True)

        self.combo_estado = QComboBox(); self.combo_estado.addItem(""); self.combo_estado.addItems(estados)
        self.combo_muni = QComboBox(); self.combo_muni.addItem("")

        self.btn_buscar = QPushButton("Buscar")
        self.btn_limpiar = QPushButton("Limpiar")
        self.btn_salir = QPushButton("Salir")

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)  # desactivado por defecto

        self.busy_label = QLabel("⏳ Buscando...")
        self.busy_label.setVisible(False)
        self.busy_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.output = QTextEdit(); self.output.setReadOnly(True)

        grid = QGridLayout()
        grid.addWidget(QLabel("Consulta"), 0, 0)
        grid.addWidget(self.input_query, 0, 1, 1, 3)
        grid.addWidget(self.check_auto_location, 1, 1, 1, 3)
        grid.addWidget(QLabel("Estado"), 2, 0)
        grid.addWidget(self.combo_estado, 2, 1)
        grid.addWidget(QLabel("Municipio"), 2, 2)
        grid.addWidget(self.combo_muni, 2, 3)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_buscar)
        btn_row.addWidget(self.btn_limpiar)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_salir)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.busy_label)
        btn_row.addWidget(self.check_fastpath)

        vbox_search = QVBoxLayout()
        vbox_search.addWidget(title)
        vbox_search.addLayout(grid)
        vbox_search.addLayout(btn_row)
        vbox_search.addWidget(QLabel("Resultados"))
        vbox_search.addWidget(self.output)

        tab_search = QWidget(); tab_search.setLayout(vbox_search)

        # ---- Tab: Supervisión ----
        sup_title = QLabel("Supervisión — Sesión actual")
        sup_title.setFont(QFont("Segoe UI", 12, QFont.Bold))

        self.lbl_total = QLabel("Consultas totales: 0")
        self.lbl_success = QLabel("Resueltas (ok): 0")
        self.lbl_empty = QLabel("Sin resultados: 0")
        self.lbl_success_rate = QLabel("Éxito: 0.0%")
        self.lbl_lat = QLabel("Latencia (p50 / p95 / avg): 0 / 0 / 0 ms")


        self.btn_export = QPushButton("Exportar telemetría (CSV)")
        self.btn_export_last = QPushButton("Exportar últimos resultados (CSV)")

        self.table_top = QTableWidget(0, 2)
        self.table_top.setHorizontalHeaderLabels(["Consulta", "Frecuencia"])
        self.table_top.horizontalHeader().setStretchLastSection(True)


        vbox_sup = QVBoxLayout()
        vbox_sup.addWidget(sup_title)
        vbox_sup.addWidget(self.lbl_total)
        vbox_sup.addWidget(self.lbl_success)
        vbox_sup.addWidget(self.lbl_empty)
        vbox_sup.addWidget(self.lbl_success_rate)   # NUEVO
        vbox_sup.addWidget(self.lbl_lat)
        vbox_sup.addWidget(self.btn_export)
        vbox_sup.addWidget(self.btn_export_last)    # NUEVO
        vbox_sup.addWidget(QLabel("Top consultas (10)"))
        vbox_sup.addWidget(self.table_top)

        tab_sup = QWidget(); tab_sup.setLayout(vbox_sup)

        # ---- Tabs container ----
        tabs = QTabWidget()
        tabs.addTab(tab_search, "Búsqueda")
        tabs.addTab(tab_sup, "Supervisión")

        central = QWidget(); layout_main = QVBoxLayout(); layout_main.addWidget(tabs); central.setLayout(layout_main)
        self.setCentralWidget(central)

    # helpers UI
    def is_fastpath_enabled(self):
        return self.check_fastpath.isChecked()

    def update_municipios(self, municipios: list[str]):
        self.combo_muni.clear(); self.combo_muni.addItem(""); self.combo_muni.addItems(municipios)

    def append_result(self, text: str):
        self.output.append(text + "\n")

    def set_busy(self, busy: bool):
        self.btn_buscar.setEnabled(not busy)
        self.btn_limpiar.setEnabled(not busy)
        self.btn_salir.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.busy_label.setVisible(busy)
        # if hasattr(self, "spinner"):
        #     self.spinner.setVisible(busy)
        #     (self.spinner_movie.start() if busy else self.spinner_movie.stop())

    def clear_all(self):
        self.input_query.clear()
        self.combo_estado.setCurrentIndex(0)
        self.combo_muni.clear(); self.combo_muni.addItem("")
        self.output.clear()

    # --- Supervisión helpers ---
    def update_metrics_view(self, snap: dict):
        self.lbl_total.setText(f"Consultas totales: {snap.get('total', 0)}")
        self.lbl_success.setText(f"Resueltas (ok): {snap.get('success', 0)}")
        self.lbl_empty.setText(f"Sin resultados: {snap.get('empty', 0)}")
        self.lbl_success_rate.setText(f"Éxito: {snap.get('success_rate', 0.0):.1f}%")
        self.lbl_lat.setText(
            f"Latencia (p50 / p95 / avg): {snap.get('lat_p50', 0):.0f} / {snap.get('lat_p95', 0):.0f} / {snap.get('avg_lat', 0):.0f} ms"
        )
        # top queries
        top = snap.get("top_queries", [])
        self.table_top.setRowCount(len(top))
        for r, (q, cnt) in enumerate(top):
            self.table_top.setItem(r, 0, QTableWidgetItem(q))
            self.table_top.setItem(r, 1, QTableWidgetItem(str(cnt)))

    def ask_export_path(self) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, "Guardar telemetría como...", "telemetria.csv", "CSV (*.csv)")
        return path or None

    def ask_export_last_hits_path(self) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, "Guardar últimos resultados como...", "ultimos_resultados.csv",
                                              "CSV (*.csv)")
        return path or None