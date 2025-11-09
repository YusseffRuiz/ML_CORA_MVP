import torch
from langchain_community.llms import CTransformers

#################################Augmentation and Generation ##########################
# Planteamiento del Modelo de LLM


class GenerationModuleLlama:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", configFile = None):
        """
        :param model_name: model name or model path
        :param high_gpu: if gpu has enough power
        """
        self.initial_prompt = None

        # Verificar GPU
        print("CUDA available:", torch.cuda.is_available())

        print("Initializing Model ...")
        if configFile is None:
            config = {'max_new_tokens': 512, 'context_length': 2000, 'temperature': 0.45}
        else:
            config = configFile

        self.llm_model = CTransformers(
            model=model_name,
            config=config,
            verbose=True,
            device=device,
        )
        print("Module Created!")
    def initialize(self, initial_prompt):
        self.initial_prompt = initial_prompt


    def build_llama2_prompt(self, context: str, question: str) -> str:
        # Plantilla oficial LLaMA-2 chat
        return (
            f"[INST] <<SYS>>\n{self.initial_prompt}\n<</SYS>>\n\n"
            f"# CONTEXTO\n{context}\n\n"
            f"# PREGUNTA\n{question}\n"
            "[/INST]"
        )

    ## Función principal de respuestas interpretadas por el sistema RAG
    def rag_answer(self, query: str, retrieval):
        # Uso de nuestra función ask previamente desarrollada. - Retrieval
        resp, docs = retrieval.ask(query, return_docs=True)
        # print("Respuesta basica: " + str(docs))
        # print("####################################################")
        if not docs:
            return "No encontré información suficiente en la base."

        # Armar contexto a partir de nuestro doc creado
        context = build_context_from_docs(docs)

        # Llamada al LLM con prompt RAG - Augmentation
        # prompt_value = initial_promtp.format(context=context, question=query)
        prompt_value = self.build_llama2_prompt(context=context, question=query)
        # Generation
        out = self.llm_model.invoke(prompt_value)

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
        m = d.metadata

        snippet = d.page_content

        # Recorta un poco por si el chunk es largo (opcional, por seguridad)
        # if len(snippet) > 900:
        #     snippet = snippet[:900] + " ..."

        # source = f"(Fuente: {m.get('id','?')} — {m.get('municipio','?')}, {m.get('estado','?')})"
        source = f"(Fuente: {m.get('id', '?')} "
        parts.append(f"{snippet}\n{source}")

    return "\n---\n".join(parts)