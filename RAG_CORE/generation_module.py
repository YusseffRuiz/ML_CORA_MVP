import re

import tiktoken
import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
from llama_cpp import Llama
import os, time
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#################################Augmentation and Generation ##########################
# Planteamiento del Modelo de LLM
def _try_tiktoken_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

_enc = _try_tiktoken_encoding()

def count_tokens(text: str) -> int:
    """Cuenta tokens usando tiktoken si está disponible; si no, aproxima por palabras."""
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    # fallback: aprox 1 token ≈ 0.75 palabras en español
    return int(len(text.split()) / 0.75) + 1


class GenerationModuleLlama:
    def __init__(self, model_name, retrieval, device="cuda" if torch.cuda.is_available() else "cpu", configFile = None):
        """
        :param model_name: model name or model path
        :param device: if gpu his available
        """
        self.memoria = None
        self.initial_prompt = None
        self.retrieval = retrieval
        self.follow_up_model = FollowUpDetector(retrieval=self.retrieval, threshold=0.5)

        # Verificar GPU
        cuda_available = True if device == "cuda" else False

        print("Initializing Model ...")
        print("CUDA available:", cuda_available)
        gpu_layers = 20
        max_tokens = 16572
        if configFile is None:
            if cuda_available:
                gpu_layers = 20
                config = {'max_new_tokens': 256, 'context_length': 1800, 'temperature': 0.45, "gpu_layers": gpu_layers,
                          "threads": os.cpu_count()}
            else:
                config = {'max_new_tokens': 256, 'context_length': 1800, 'temperature': 0.45, "threads": os.cpu_count()}
        else:
            config = configFile


        self.llm_model = Llama(model_path=model_name,
                         n_ctx=max_tokens,
                         # The max sequence length to use - note that longer sequence lengths require much more resources
                         n_threads=config["threads"],
                         # The number of CPU threads to use, tailor to your system and the resulting performance
                         n_gpu_layers=gpu_layers,
                         temperature=config["temperature"],
                         use_mlock=True,
                         verbose=False
                         )
        self.memoria = ConversationalMemory(max_tokens=max_tokens)
        print(f"Module Created!, gpu layers: {gpu_layers}.")
    def initialize(self, initial_prompt):
        self.initial_prompt = initial_prompt


    def build_llama2_prompt(self, context: str, question: str, historial: str = None) -> str:
        # Plantilla oficial LLaMA-2 chat
        if historial is not None:
            return (
                f"[INST] <<SYS>>{self.initial_prompt}<< / SYS >>"
                f"# HISTORIAL: {historial}"
                f"# CONTEXTO: {context}"
                f"# PREGUNTA: {question}[/ INST]"
            )
        else:
            return (
                f"[INST] <<SYS>>\n{self.initial_prompt.strip()}\n<</SYS>>\n\n"
                f"# CONTEXTO\n{context.strip()}\n\n"
                f"# PREGUNTA\n{question.strip()}\n[/INST]\n"
            )


    ## Función principal de respuestas interpretadas por el sistema RAG
    def rag_answer(self, query: str, debug : bool = False):
        """
        Regresa la respuesta del sistema LLM y un bool indicando si ya termino la interaccion (True) o si continua (False).
        """
        # Uso de nuestra función ask previamente desarrollada. - Retrieval
        query = query.lower()

        # 7. Detección simple de fin de sesión
        if self.follow_up_model.detect_exit_intent(query):
            self.memoria.clear()
            return "🤖Ha sido un gusto ayudarte. ¡Que tengas buen día! ¡Adiós!", True

        docs = self.memoria.get_last_docs() # revisamos si hay informacion previa
        used_cached_docs = docs is not None and len(docs) > 0

        follow_up = self.follow_up_model.is_follow_up(query, self.memoria.get_recent_turns())
        # Verificación adicional: ¿el usuario cambió de intención a pesar de ser follow-up?
        if follow_up and not self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns()):
            if debug:
                print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
            follow_up = False  # Forzamos modo nueva búsqueda
            docs = []  # Limpiamos docs previos para no arrastrar contexto viejo
        if debug:
            print("Follow up: ", follow_up)

        if not follow_up:
            # Validamos si el usuario sigue en la misma intención, aunque no sea un follow-up directo
            if self.should_continue_context(query, used_cached_docs):
                follow_up = True
                if debug:
                    print(
                        "[DEBUG] No se detectó follow-up directo, pero sí continuidad semántica. Se mantiene contexto.")
            else:
                resp, docs = self.retrieval.ask(query, return_docs=True, memoria=None)

                if not docs:
                    self.memoria.add_turn("user", query)
                    self.memoria.add_turn("assistant", "No encontré información suficiente en la base.")
                    return "No encontré información suficiente en la base.", False

                self.memoria.set_last_docs(docs)
                if debug:
                    print("[DEBUG] Nueva búsqueda realizada, se guardan nuevos documentos.")
                self.initial_prompt = self.INIT_PROMPT_LLAMA

        else:
            # 4. En caso de follow-up, usar los documentos previos
            if not used_cached_docs:
                if debug:
                    print("[DEBUG] Follow-up detectado, pero no hay documentos previos.")
                self.memoria.add_turn("user", query)
                self.memoria.add_turn("assistant", "No tengo contexto previo suficiente.")
                return "¿Podrías especificar a qué unidad o tema te refieres?", False

            matches = self.follow_up_model.match_sucursal_from_input(user_input=query, docs=docs)
            if matches:
                docs = [doc for doc, _ in matches]
                self.memoria.set_last_docs(docs)
                self.initial_prompt = self.DETAILS_PROMPT
                if debug:
                    print(f"[DEBUG] Se detectaron {len(matches)} coincidencias por follow-up:")
                    for doc in docs:
                        nombre = doc.metadata.get("nombre_oficial", "Sin nombre")
                        municipio = doc.metadata.get("municipio", "Sin municipio")
                        estado = doc.metadata.get("estado", "Sin estado")
                        print(f" - {nombre} ({municipio}, {estado})")
            else:
                # Solo considerar cambio de intención si NO hubo match
                is_same_topic = self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns())
                if debug:
                    print(f"[DEBUG] Similitud semántica entre queries: {is_same_topic}")
                if not is_same_topic:
                    if debug:
                        print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
                    follow_up = False
                    docs = []
                    self.memoria.set_last_docs([])
                else:
                    self.initial_prompt = self.CONTINUOUS_PROMPT_LLAMA
                    if debug:
                        print("[DEBUG] Follow-up válido sin coincidencia específica. Se mantiene el contexto actual.")

        context = build_context_from_docs(docs, full=follow_up)

        historial = "\n".join([f"{t['role'].title()}: {t['content']}" for t in self.memoria.get_recent_turns()])
        context_tokens = count_tokens(context)
        historial_tokens = count_tokens(historial)
        self.memoria.trim_if_exceeds_tokens(context_tokens, historial_tokens)

        # Armado de prompt
        prompt_value = self.build_llama2_prompt(context=context, question=query, historial=historial)

        if debug:
            print("Finished Context, tokens number: ", count_tokens(prompt_value))

        # Generation
        t0 = time.perf_counter()
        out = self.llm_model(
            prompt=prompt_value,
            stop=["</s>"],
            max_tokens=512,
            echo=False,
            stream=False
        )
        t1 = time.perf_counter()

        # Actualizar memoria
        self.memoria.add_turn("user", query)
        self.memoria.add_turn("assistant", out)

        ## Debugging
        if debug:
            print("Finished invoke, time: ", t1 - t0, " s")
            print("Prompt completo:\n", prompt_value)
            # print("Respuesta tokens:", len(out["choices"][0]["text"].split()))
            # print("Respuesta completa:\n", out["choices"][0]["text"])

        try:
            out_text = out["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[ERROR] No se pudo extraer texto de la salida del modelo: {e}")
            out_text = resp.get("answer", "") or str(out)  # último recurso

        return out_text, False


    def should_continue_context(self, query: str, used_cached_docs: bool) -> bool:
        """
        Determina si se debe mantener el contexto anterior aunque no se haya detectado un follow-up directo.
        Usa is_follow_up_user como respaldo semántico.
        """
        if not used_cached_docs:
            return False

        try:
            return self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns())
        except Exception as e:
            print(f"[DEBUG] Error al validar continuidad de contexto: {e}")
            return False

    DETAILS_PROMPT = """
    Eres un asistente telefónico en español mexicano de Medical Life. 
    El usuario ha pedido detalles de una unidad médica específica.

    Responde usando SOLO la información del contexto. No inventes.
    
    Incluye claramente:
    - Nombre oficial
    - Dirección
    - Horarios
    - Teléfonos
    - Servicios
    - Información adicional si existe

    Estilo:
    - Español neutro, respuesta clara y breve (4-6 líneas).
    - Tu respuesta debes escribirla solamente con letras, sin guiones o identificador, los números expresados con palabras.
    - NO te presentes. NO saludes. Inicia con una frase de confirmación como "claro!" y ve directo a los datos.
    - Finaliza preguntando si desea más información de alguna de las sedes o si desea terminar la comunicación.
    """

    CONTINUOUS_PROMPT_LLAMA = """
    Eres un asistente telefónico en español mexicano de Medical Life, empresa proveedora de servicios médicos.
    Responde SOLO usando el CONTEXTO otorgado. No inventes nada que no esté en el contexto.
    Si falta información, di: "No tenemos información de lo que estás pidiendo".

    Ya estas respondiendo de mensajes previos, por lo tanto no te presentes y ve directo a la respuesta, que incluya:
    - Un parafraseo corto de la pregunta del cliente.
    - Solo las sedes que se preguntaron y que se encuentren en el contexto, indicando la información pedida por el usuario.

    Instrucciones de estilo:
    - NO te presentes. NO saludes. Ve directo a los datos.
    - Responde solamente en español.
    - Si solo hay una sede, preséntala directamente, sin mencionar que no hay más.
    - NO inventes nombres de sedes ni menciones genéricas como "otras sedes disponibles".
    - La respuesta debe ser breve (3 a 6 líneas).
    - Tu respuesta debes escribirla solamente con letras, sin guiones o identificador, los números expresados con palabras.
    - Finaliza siempre preguntando si desea más información de alguna de las sedes o si desea terminar la comunicación.
    """
    INIT_PROMPT_LLAMA = """
                        Eres un asistente telefónico en español mexicano llamado CORA, de Medical Life, empresa proveedora de servicios médicos.
                        Responde SOLO usando el CONTEXTO otorgado. No inventes nada que no esté en el contexto.
                        Si falta información, di: "No tenemos información de lo que estás pidiendo".

                        Tu respuesta debe incluir:
                        - Una frase de confirmación como "por supuesto".
                        - Un parafraseo corto de la pregunta del cliente.
                        - Solo las sedes encontradas en el contexto, indicando municipio, estado y el servicio solicitado.
                        - La o las sedes expresalas en palabras simples de leer y con la ubicacion corta.

                        Instrucciones de estilo:
                        - No incluyas presentación o tu nombre, ya que es un seguimiento de la misma conversación.
                        - Responde solamente en español.
                        - Si hay varias sedes, enuméralas como lista (máximo 4).
                        - Si solo hay una sede, preséntala directamente, sin mencionar que no hay más.
                        - NO inventes nombres de sedes ni menciones genéricas como "otras sedes disponibles".
                        - Finaliza siempre preguntando si desea más información de alguna de las sedes(como horarios o ubicación exacta).
                        - La respuesta debe ser breve (3 a 6 líneas).
                        - Tu respuesta debes escribirla solamente con letras, sin guiones o identificador, los números expresados con palabras.
                        - Al mencionar las sedes, menciona el nombre y ubicacion corta, no incluyas el id o guiones.
                        """

def build_context_from_docs(docs, full=False):
    parts = []
    for d in docs:
        m = d.metadata
        if full:
            snippet = d.page_content
            parts.append(f"{snippet}. ")
        else:
            parts.append(f"{m['id']} — {m['municipio']}, {m['estado']} | Servicios: {', '.join(m.get('servicios_lista', []))}")
    return "\n---\n".join(parts)


class ConversationalMemory:
    """
    Clase dedicada a almacenar una conversacion en formato:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Metodos
    - Añadir nuevos turnos

    - Obtener el historial más reciente (get_recent_turns(n))

    - Limpiar el historial (al cerrar sesión)

    - Guardar el último set de documentos entregados por el retrieval (para follow-ups)

    """
    def __init__(self, max_tokens=4096):
        """
        max_turns: cantidad de turnos de ida y vuelta a mantener (user + assistant)
        """
        self.turns = []
        self.last_docs = []
        # self.max_turns = max_turns
        self.last_user_query = []
        self.last_assistant_query = []
        self.max_tokens = max_tokens

    def add_turn(self, role, content):
        """Agrega un nuevo mensaje al historial"""
        self.turns.append({"role": role, "content": content})
        if role == "user":
            self.last_user_query = content
        elif role == "assistant":
            self.last_assistant_query = content
        # if len(self.turns) > self.max_turns * 2:  # user + assistant = 2 turnos por ciclo
        #     self.turns = self.turns[-self.max_turns * 2:]

    def get_recent_turns(self):
        """Retorna los turnos recientes (user + assistant alternados)"""
        return self.turns

    def set_last_docs(self, docs):
        """Guarda los documentos del retrieval más reciente"""
        self.last_docs = docs

    def get_last_docs(self):
        return self.last_docs

    def clear(self):
        """ Limpia memoria e historial"""
        self.turns = []
        self.last_docs = []

    def trim_if_exceeds_tokens(self, tokens_so_far: int, tokens_next_context: int):
        """
        Recorta últimos turnos si la suma actual + contexto futuro excede el 90% del límite de tokens.

        Parámetros:
            - tokens_so_far: tokens del historial actual
            - tokens_next_context: tokens esperados del siguiente contexto
            - max_tokens: límite total aceptable del modelo
        """
        total_expected = tokens_so_far + tokens_next_context
        threshold = int(self.max_tokens * 0.9)

        if total_expected >= threshold and len(self.turns) >= 2:
            # Borra los dos turnos más antiguos (usuario y asistente)
            self.turns = self.turns[2:]


class FollowUpDetector:
    def __init__(self, retrieval=None, threshold=0.50):
        if retrieval is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.model = retrieval
        self.threshold = threshold

    def is_follow_up(self, user_input: str, memory_turns: list) -> bool:
        if not memory_turns:
            return False

        # 1. Heurístico: frases típicas de seguimiento
        heuristics = [
            r"\bs[ií]\b", r"\besa\b", r"\bla de\b", r"\bquiero más info\b",
            r"\bde la\b", r"\besa sede\b", r"\bhorario\b", r"\bdirección\b",
            r"\bdónde está\b", r"\bteléfono\b", r"\bubicación\b", r"\bcuál es el horario\b",
            r"\bme repites\b", r"\bdónde queda\b"
        ]
        for pattern in heuristics:
            if re.search(pattern, user_input):
                return True

        # 2. Última respuesta del asistente (más relevante que la pregunta anterior del usuario)
        last_assistant = next((t["content"] for t in reversed(memory_turns) if t["role"] == "assistant"), None)
        if not last_assistant:
            return False

        # 3. Similaridad semántica entre nuevo input y última respuesta
        embedding_a = self.model.embeddings.embed_query(user_input)
        last_assistant_text = (
            last_assistant["content"]["choices"][0]["text"]
            if isinstance(last_assistant, dict) and isinstance(last_assistant.get("content"), dict)
            else last_assistant.get("content") if isinstance(last_assistant, dict)
            else str(last_assistant)
        )
        print("Last text: ", last_assistant_text)
        embedding_b = self.model.embeddings.embed_query(str(last_assistant_text))
        sim = cosine_similarity([embedding_a], [embedding_b])[0][0]

        return sim >= self.threshold

    def is_follow_up_user(self, user_input: str, memory_turns: list) -> bool:
        if not memory_turns:
            return False

            # Heurístico básico (puedes dejarlo, pero como fallback)
        heuristics = [r"\bs[ií]\b", r"\besa\b", r"\bla de\b", r"\bde la\b", r"\besa sede\b"]
        if any(re.search(h, user_input.lower()) for h in heuristics):
            return True

        # Obtener última pregunta del usuario
        last_user_query = None
        for turn in reversed(memory_turns):
            if turn["role"] == "user":
                last_user_query = turn["content"]
                break

        if not last_user_query:
            return False

        # Comparar embeddings
        emb_current = self.model.embeddings.embed_query(user_input)
        emb_previous = self.model.embeddings.embed_query(last_user_query)
        sim = cosine_similarity([emb_current], [emb_previous])[0][0]

        print(f"[DEBUG] Similitud semántica entre queries: {sim:.3f}")

        return sim >= self.threshold  # threshold por defecto: 0.65–0.75

    def match_sucursal_from_input(self, user_input: str, docs: list, threshold: float = 0.72, top_k: int = 1):
        """
        Dada una entrada del usuario y una lista de documentos (sucursales),
        devuelve la(s) sucursal(es) que más se parezcan con base en embeddings.

        Parámetros:
            - user_input: texto del usuario (ej. "de la Gustavo Madero")
            - docs: lista de Document con metadatos (output de last_docs)
            - encoder: Usa el vectorstore y embeddings del sistema.
            - threshold: umbral mínimo de similitud
            - top_k: cuántas coincidencias devolver (por default solo 1)

        Retorna:
            Lista de Document (las mejores coincidencias)
        """
        if not docs:
            return []

            # Paso 1: preparar corpus reducido (texto representativo por doc)
        temp_docs = []
        for d in docs:
            m = d.metadata
            # Concatenamos nombre oficial + municipio + estado (ajustable)
            doc_text = f"{m.get('nombre_oficial', '')} {m.get('municipio', '')} {m.get('estado', '')}"
            # Creamos Document temporal con mismo ID y score dummy
            temp_docs.append(Document(page_content=doc_text, metadata=m))

        # Paso 2: construir FAISS temporal en memoria con solo estos docs
        faiss_local = FAISS.from_documents(temp_docs, embedding=self.model.embeddings)

        # Paso 3: búsqueda en el vectorstore temporal
        results = faiss_local.similarity_search_with_score(user_input, k=top_k)
        matches = [(doc.metadata, score) for doc, score in results if score >= threshold]

        # Paso 4: devolver el documento original completo
        docs_out = []
        for meta, score in matches:
            for d in docs:
                if d.metadata.get("id") == meta.get("id"):
                    docs_out.append((d, score))
                    break

        return docs_out

    def detect_exit_intent(self, user_input: str) -> bool:
        # heurístico rápido
        exit_patterns = [r"\b(terminar|adiós|eso es todo|cortar|no tengo más preguntas|terminar|es todo|seria todo)\b"]
        if any(re.search(p, user_input.lower()) for p in exit_patterns):
            return True

        # comparación semántica
        ejemplos_salida = [
            "ya terminé", "eso es todo", "puedes cortar la llamada", "terminamos", "no tengo mas preguntas", "adiós",
            "es todo", "seria todo",
        ]
        emb_user = self.model.embeddings.embed_query(user_input)
        emb_refs = [self.model.embeddings.embed_query(e) for e in ejemplos_salida]
        scores = [cosine_similarity([emb_user], [e])[0][0] for e in emb_refs]
        print("[DEBUG] Nivel de deteccion de finalizacion: ", scores)
        return max(scores) >= 0.58  # umbral ajustable