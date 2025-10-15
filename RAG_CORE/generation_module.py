import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

#################################Augmentation and Generation ##########################
# Planteamiento del Modelo de LLM


class GenerationModule:
    def __init__(self, model_name, high_gpu=False):
        """
        :param model_name: model name or model path
        :param high_gpu: if gpu has enough power
        """
        self.high_cpu = high_gpu
        self.initial_prompt = None

        # Verificar GPU
        print("CUDA available:", torch.cuda.is_available())

        print("Initializing Model ...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Initialize model

        # Si hay poco poder o nula GPU
        if not self.high_cpu:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

         # Opción A (GPU sobrada): pesos en bf16/fp16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,  # Disable si usamos una GPU decente
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        #Usamos función pipeline de la biblioteca Transformers para generar la respuesta en función del prompt.
        gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            # parámetros conservadores para RAG
            max_new_tokens=150,
            do_sample=False,         # determinista (evita desvíos)
            temperature=0.0,
            return_full_text=False
        )

        print("Generating LLM ...")

        # Adaptador LangChain
        self.llm = HuggingFacePipeline(pipeline=gen)



    def create_initial_prompt(self, prompt):
        # Prompt inicial para evitar alucionaciones y tomar el papel adecuado.
        # Augmentation Portion

        self.initial_prompt = ChatPromptTemplate.from_template(prompt)

    ## Función principal de respuestas interpretadas por el sistema RAG
    def rag_answer(self, retrieval_module, query: str):
        # Uso de nuestra función ask previamente desarrollada. - Retrieval
        docs = retrieval_module.ask(query, return_docs=True)
        if not docs:
            return "No encontré información suficiente en la base."

        # Armar contexto a partir de nuestro doc creado
        context = build_context_from_docs(docs)

        # Llamada al LLM con prompt RAG - Augmentation
        prompt_value = self.initial_prompt.format(context=context, question=query)
        # Generation
        out = self.llm.invoke(prompt_value)

        # Guardrail de salida: validar que cite al menos un ID presente en el contexto
        ctx_ids = {d.metadata.get("id") for d in docs}
        out_text = out if isinstance(out, str) else str(out)
        if not any(idv and idv in out_text for idv in ctx_ids):
            # Fallback seguro: resume con tu `ask()` clásico
            resp = retrieval_module.ask(query, top_k=10)
            return resp["answer"]

        return out_text.strip()

def build_context_from_docs(docs):
    # Insertar SOLO lo necesario. Usar page_content.
    parts = []
    for d in docs:
        m = d.metadata

        snippet = d.page_content

        # Recorta un poco por si el chunk es largo (opcional, por seguridad)
        if len(snippet) > 900:
            snippet = snippet[:900] + " ..."

        source = f"(Fuente: {m.get('id','?')} — {m.get('municipio','?')}, {m.get('estado','?')})"
        parts.append(f"{snippet}\n{source}")

    return "\n---\n".join(parts)