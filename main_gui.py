import os, sys
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication
from RAG_CORE.retrieval_module import RetrievalModule
from UI.view import MainWindow
from UI.controller import AppController
from UI.telemetria import TelemetryLogger, SessionMetrics

def main():
    load_dotenv(os.path.expanduser("~/Documents/tokens.env"))
    HF_TOKEN = os.getenv("HF_TOKEN")

    EXCEL_PATH = os.path.join("Documents", "medical_life_real.xlsx")
    # MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"
    MODEL_NAME = "intfloat/multilingual-e5-base" # second Option, compare options

    retrieval = RetrievalModule(database_path=EXCEL_PATH, hf_token=HF_TOKEN, model_name=MODEL_NAME)
    retrieval.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold = 0.34, percentile = 0.9)
    kb_version = getattr(retrieval, "kb_version", "")  # opcional si lo expones
    model_name = getattr(retrieval, "model_name", MODEL_NAME)

    # Telemetría
    os.makedirs("logs", exist_ok=True)
    tlog = TelemetryLogger(csv_path=os.path.join("logs", "queries.csv"))
    smetrics = SessionMetrics()

    app = QApplication(sys.argv)
    estados = sorted(set(str(x) for x in retrieval.kb_df["estado"].dropna().unique()))
    win = MainWindow(estados=estados)

    controller = AppController(
        win, retrieval, retrieval.kb_df,
        on_query_start=None, on_query_end=None,
        telemetry_logger=tlog, session_metrics=smetrics,
        kb_version=kb_version, model_name=model_name
    )

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
