import tiktoken
import torch
# from langchain_community.llms import CTransformers
from llama_cpp import Llama
import os, time

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
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", configFile = None):
        """
        :param model_name: model name or model path
        :param high_gpu: if gpu has enough power
        """
        self.initial_prompt = None

        # Verificar GPU
        cuda_available = True if device == "cuda" else False

        print("Initializing Model ...")
        print("CUDA available:", cuda_available)
        gpu_layers = 20
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
                         n_ctx=4096,
                         # The max sequence length to use - note that longer sequence lengths require much more resources
                         n_threads=config["threads"],
                         # The number of CPU threads to use, tailor to your system and the resulting performance
                         n_gpu_layers=gpu_layers,
                         temperature=config["temperature"],
                         use_mlock=True,
                         verbose=False
                         )

        # self.llm_model = CTransformers(
        #     model=model_name,
        #     model_type="llama",
        #     config=config,
        #     verbose=False,
        #     device=device,
        #     gpu_layers=gpu_layers,
        # )
        print(f"Module Created!, gpu layers: {gpu_layers}.")
    def initialize(self, initial_prompt):
        self.initial_prompt = initial_prompt


    def build_llama2_prompt(self, context: str, question: str) -> str:
        # Plantilla oficial LLaMA-2 chat
        # print("Tokens in prompt: ", count_tokens(self.initial_prompt))
        # print("Tokens in context: ", count_tokens(context))
        return (
            f"[INST] <<SYS>>\n{self.initial_prompt.strip()}\n<</SYS>>\n\n"
            f"# CONTEXTO\n{context.strip()}\n\n"
            f"# PREGUNTA\n{question.strip()}\n[/INST]\n"
        )


    ## Función principal de respuestas interpretadas por el sistema RAG
    def rag_answer(self, query: str, retrieval):
        # Uso de nuestra función ask previamente desarrollada. - Retrieval
        resp, docs = retrieval.ask(query, return_docs=True)
        # print("Respuesta basica: " + str(docs))
        print("####################################################")
        if not docs:
            return "No encontré información suficiente en la base."

        # Armar contexto a partir de nuestro doc creado
        context = build_context_from_docs(docs)
        # print(context)

        # Llamada al LLM con prompt RAG - Augmentation
        # prompt_value = initial_promtp.format(context=context, question=query)
        prompt_value = self.build_llama2_prompt(context=context, question=query)
        print("Finished Context, tokens number: ", count_tokens(prompt_value))
        # Generation
        t0 = time.perf_counter()
        # out = self.llm_model.invoke(prompt_value)
        out = self.llm_model(
            prompt=prompt_value,
            stop=["</s>"],
            max_tokens=512,
            echo=False,
            stream=False
        )
        t1 = time.perf_counter()

        print("Finished invoke, time: ", t1 - t0, " s")
        print("Prompt completo:\n", prompt_value)
        print("Respuesta tokens:", len(out["choices"][0]["text"].split()))
        print("Respuesta completa:\n", out["choices"][0]["text"])

        out_text = out["choices"][0]["text"].strip() if isinstance(out, dict) else str(out)

        if not out_text:
            return resp["answer"]

        return out_text.strip()

        # return out
        # Guardrail de salida: validar que cite al menos un ID presente en el contexto
        # ctx_ids = {d.metadata.get("id") for d in docs}
        out_text = out if isinstance(out, str) else str(out)
        if not out_text:
            return resp["answer"]

        return out_text.strip()

def build_context_from_docs(docs):
    # Insertar SOLO lo necesario. Usar page_content.
    parts = []
    for d in docs:
        snippet = d.page_content
        parts.append(f"{snippet}. ")

    return "\n---\n".join(parts)