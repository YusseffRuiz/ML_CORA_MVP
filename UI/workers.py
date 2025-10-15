# -*- coding: utf-8 -*-
from PySide6.QtCore import QObject, Signal, QRunnable, Slot
import traceback
import time

class AskWorkerSignals(QObject):
    finished = Signal(dict)  # {text, tokens, latency_ms, cancelled}
    progress = Signal(str)  # streaming tokens
    error = Signal(str)

class AskWorker(QRunnable):
    def __init__(self, retrieval, query: str, filtros: dict, job_id: int, llm_model=None):
        super().__init__()
        self.retrieval = retrieval
        self.llm_model = llm_model
        self.query = query
        self.job_id = job_id
        self.filtros = filtros
        self.signals = AskWorkerSignals()

    @Slot()
    def run(self):
        try:
            if self.llm_model is not None:
                resp = self.llm_model.rag_answer(self.query, self.retrieval)  # Faster
            else:
                resp = self.retrieval.ask(self.query)
            # empaqueta job_id para que el controller sepa si aún es válido
            self.signals.finished.emit({"_job_id": self.job_id, **resp})
        except Exception:
            self.signals.error.emit(traceback.format_exc())



# Worker base
class GenWorker(QObject):

    def __init__(self, llm, context, question, max_tokens=300):
        super().__init__()
        self._stop = False
        self.llm = llm
        self.ctx = context
        self.q = question
        self.max_tokens = max_tokens
        self.signals = AskWorkerSignals()

    @Slot()
    def run(self):
        try:
            t0 = time.perf_counter()
            buf = []
            # llama-cpp: stream
            for chunk in self.llm.llm.create_chat_completion(
                messages=build_messages(self.ctx, self.q),
                max_tokens=self.max_tokens,
                temperature=0.0,
                stream=True
            ):
                if self._stop:
                    self.signals.finished.emit({"text":"(cancelado)", "tokens":len(buf), "latency_ms": int((time.perf_counter()-t0)*1000), "cancelled": True})
                    return
                token = chunk["choices"][0]["delta"].get("content","")
                if token:
                    buf.append(token)
                    self.signals.progress.emit(token)
            txt = "".join(buf)
            self.signals.finished.emit({"text": txt, "tokens": len(buf), "latency_ms": int((time.perf_counter()-t0)*1000), "cancelled": False})
        except Exception as e:
            self.signals.error.emit(str(e))

    def stop(self):
        self._stop = True
