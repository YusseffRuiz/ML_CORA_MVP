from llama_cpp import Llama
import os
from dotenv import load_dotenv

INIT_PROMPT_LLAMA = """
                    Eres un asistente telefónico en español mexicano de Medical Life, empresa proveedora de servicios médicos.
                    Responde SOLO usando el CONTEXTO otorgado. No inventes nada que no esté en el contexto.
                    Si falta información, di: "No tenemos información de lo que estás pidiendo".

                    Tu respuesta debe incluir:
                    - Un saludo breve, presentándote como "Cora, Asistente Telefónico de Medical Life".
                    - Un parafraseo corto de la pregunta del cliente.
                    - Solo las sedes encontradas en el contexto, indicando municipio, estado y solo el servicio solicitado.
                    - La o las sedes expresalas en palabras simples de leer y con la ubicacion corta.

                    Instrucciones de estilo:
                    - Responde solamente en español.
                    - Si hay varias sedes, enuméralas como lista (máximo 4) con nombre, municipio, estado y el servicio que se pidió.
                    - Si solo hay una sede, preséntala directamente, sin mencionar que no hay más.
                    - NO inventes nombres de sedes ni menciones genéricas como "otras sedes disponibles".
                    - Finaliza siempre preguntando si desea más información de alguna de las sedes(como horarios o ubicación exacta).
                    - La respuesta debe ser breve (3 a 6 líneas).
                    - Al mencionar las sedes, menciona el nombre y ubicacion corta, no incluyas el id o guiones.
                    - Tu respuesta debes escribirla solamente con letras, sin guiones o identificador, los números expresados con palabras.
                    """
LLM_MODEL_FILE = "../HF_Agents/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

load_dotenv("Documents/tokens.env")  # busca .env en el cwd
HF_TOKEN = os.getenv("HF_TOKEN")

MEDICAL_EXTENDED = "Documents/medical_life_real.xlsx"

VECTOR_MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
gpu_layers = 20
config = {'max_new_tokens': 256, 'context_length': 1800, 'temperature': 0.45, "gpu_layers": gpu_layers,
                          "threads": os.cpu_count()}
llm_model = Llama(model_path=LLM_MODEL_FILE,
                         n_ctx=config["context_length"],
                         # The max sequence length to use - note that longer sequence lengths require much more resources
                         n_threads=config["threads"],
                         # The number of CPU threads to use, tailor to your system and the resulting performance
                         n_gpu_layers=gpu_layers,
                         temperature=config["temperature"],
                         use_mlock=False,
                         use_mmap=True,
                         verbose=False
                         )

from RAG_CORE.generation_module import GenerationModuleLlama
from RAG_CORE.retrieval_module import RetrievalModule
from AudioTranscription.ASREngine import AsrEngine
from AudioTranscription.audio_recording import AudioRecorder

# Importar llaves de uso

from TTS_test import Speaker

import time
from pydub import AudioSegment
from pydub.playback import play



#  Voz de saludo
def play_start_sound(speaker=None):
    speaker.speak("¡Hola! Mi nombre es Cora, soy el asistente de Medical Laif para resolver tus dudas y es un placer atender tu llamada."
                  "Dime, ¿tienes alguna duda sobre un servicio o localización de algún centro?")

def receive_question(asr=None, recorder = None):
    # 1. Grabar hasta detectar silencio
    if recorder is not None:
        raw_audio = recorder.record_until_silence()

        if raw_audio is None or raw_audio.size == 0:
            print("No se detectó nada, esperando...")
            return None

        # 2. Guardar audio temporalmente
        tmp_path = recorder.save_audio(raw_audio)

        # 3. Transcribir
        pregunta, _, _ = asr.transcribe_file(tmp_path, language="es")
        os.remove(tmp_path)
    else:
        pregunta = input("👤 Usuario: ").strip()
    return pregunta

def flow(asr=None, speaker=None, llm_model=None, recorder=None):
    if speaker is not None:
        play_start_sound(speaker=speaker)
    while True:
        try:
           pregunta = receive_question(asr=asr, recorder=recorder)
           if pregunta.strip():
               respuesta, finish_flag = llm_model.rag_answer(query=pregunta)
               print("🤖 Cora:", respuesta)
               if speaker is not None:
                   # speaker.speak(respuesta)
                    path = speaker.save_dialog(text=respuesta.text)
                    speaker.speak_from_path(path)
               print("-" * 50)

               # Esperar nueva pregunta
               if "adios" in respuesta.lower() or finish_flag:
                   break
           else:
               print("😶 No se entendió lo que dijiste.")
               # speaker.speak("No entendí, ¿podrías repetirme tu pregunta? ")
               interaction = "No entendí, ¿podrías repetirme tu pregunta? "
               speaker.save_dialog(text=interaction)
               speaker.speak_from_path(path)

        except KeyboardInterrupt:
            print("\n👋 Conversación terminada.")
            break


def main():

    if not HF_TOKEN:
        print("HF_TOKEN no encontrado")
        exit(0)

    retrieval_module = RetrievalModule(database_path=MEDICAL_EXTENDED, hf_token=HF_TOKEN, model_name=VECTOR_MODEL_NAME)
    retrieval_module.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold = 0.34, percentile = 0.9)

    """
    Augmentation and Generation Portion
    """
    llm_module = GenerationModuleLlama(llm_model)
    llm_module.initialize(initial_prompt=INIT_PROMPT_LLAMA, retrieval=retrieval_module, debug=True)

    """
    Audio Initialization
    """
    speaker = Speaker(engine="KOKORO")
    print("Speaker Initialized")
    asr = AsrEngine(model_size="medium", device="cuda")
    print("ASR Initialized")
    recorder = AudioRecorder()
    print("Recorder Initialized")
    flow(asr=asr, speaker=speaker, llm_model=llm_module, recorder=recorder)

def test_mic():
    # Inicializa grabador
    import sounddevice as sd
    recorder = AudioRecorder()
    asr = AsrEngine(model_size="medium", device="cuda")

    final_audio = recorder.record_until_silence()

    # Guardar y reproducir
    tmp_path = recorder.save_audio(final_audio)

    print("🔁 Reproduciendo lo que grabaste...")
    sound = AudioSegment.from_wav(tmp_path)
    play(sound)

    # Transcripción
    texto, _, _ = asr.transcribe_file(tmp_path, language="es")
    print("📝 Transcripción:", texto)

    # Limpieza
    time.sleep(1)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print("🧹 Archivo temporal eliminado")
    else:
        print("⚠️ No se encontró el archivo para eliminar")


if __name__ == "__main__":
    main()
    # test_mic()