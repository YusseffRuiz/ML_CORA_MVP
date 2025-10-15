import os, sys
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication
from RAG_CORE.retrieval_module import RetrievalModule
from UI.view import MainWindow
from UI.controller import AppController
from UI.telemetria import TelemetryLogger, SessionMetrics
from RAG_CORE.generation_module import GenerationModuleLlama

INIT_PROMPT_LLAMA = """
                    Eres un asistente telefónico de Medical Life, empresa proveedora de servicios medicos. Responde SOLO usando el CONTEXTO.
                    No inventes nada que no esté en el contexto. Si falta información, di "No tenemos información de lo que estas pidiendo".
                    Incluye en la respuesta las sedes relevantes (nombre de la unidad, Municipio, Estado, Teléfono Tipo y Programa al que pertenece), los servicios que ofrecen, los horarios de atención, si tiene requisitos o si hay alguna otra nota regla.

                    Instrucciones de estilo:
                    - Responde breve (3–6 líneas).
                    - Si hay varias opciones, preséntalas como lista con 4 ítems máximo, sin repetir informacion. 
                    Por ejemplo: ... algunas opciones, donde ofrecemos una amplia gama de pruebas clinicas, incluyendo
                    tal tal y tal, en Sede 1, Sede 2 y Sede 3.
                    - Da seguimiento por si la persona quiere mas informacion.
                    """


def main():
    load_dotenv(os.path.expanduser("~/Documents/tokens.env"))
    HF_TOKEN = os.getenv("HF_TOKEN")

    EXCEL_PATH = os.path.join("Documents", "medical_life_real.xlsx")
    MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"
    # MODEL_NAME = "intfloat/multilingual-e5-base" # second Option, compare options

    retrieval = RetrievalModule(database_path=EXCEL_PATH, hf_token=HF_TOKEN, model_name=MODEL_NAME)
    retrieval.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold = 0.34, percentile = 0.9)
    kb_version = getattr(retrieval, "kb_version", "")  # opcional si lo expones
    model_name = getattr(retrieval, "model_name", MODEL_NAME)

    llm_model_file = "llama-2-7b-chat.Q4_K_M.gguf"
    llm_module = GenerationModuleLlama(llm_model_file)
    llm_module.initialize(initial_prompt=INIT_PROMPT_LLAMA)

    # Telemetría
    os.makedirs("logs", exist_ok=True)
    tlog = TelemetryLogger(csv_path=os.path.join("logs", "queries.csv"))
    smetrics = SessionMetrics()

    app = QApplication(sys.argv)
    estados = sorted(set(str(x) for x in retrieval.kb_df["estado"].dropna().unique()))
    win = MainWindow(estados=estados)

    controller = AppController(
        win, llm_model=llm_module, retrieval_module=retrieval, kb_df=retrieval.kb_df,
        on_query_start=None, on_query_end=None,
        telemetry_logger=tlog, session_metrics=smetrics,
        kb_version=kb_version, model_name=model_name
    )

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
