import torch
import torchaudio
from TTS.utils.radam import RAdam
import torch.serialization
from TTS.api import TTS

from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd


from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf


# model_name = "tts_models/es/mai/tacotron2-DDC"
# model_name = "tts_models/en/ljspeech/glow-tts"
model_name = "tts_models/es/css10/vits"
# model_name = "tts_models/multilingual/multi-dataset/your_tts"
kokoro_voice = 'ef_dora'


class Speaker:
    def __init__(self, engine="TTS", device= "cuda" if torch.cuda.is_available() else "cpu"):
        self.engine = engine
        if engine == "TTS":
            torch.serialization.add_safe_globals([RAdam])
            self.tts = TTS(model_name).to(device)
        elif engine == "KOKORO":
            self.tts = KPipeline(lang_code='e')
        else:
            print("Only TTS and KOKORO are supported")

        self.saving_count = 0

    def speak(self, text):
        if self.engine == "TTS":
            wav = self.tts.tts(text)
            sd.play(wav, samplerate=22050)
            sd.wait()
        elif self.engine == "KOKORO":
            generator = self.tts(text, voice=kokoro_voice)
            for i, (gs, ps, audio) in enumerate(generator):
                audio = audio.detach().cpu().numpy()
                sd.play(audio, samplerate=24000)
                sd.wait()
        else:
            print("Not Supported Engine")

    def save_dialog(self, text, path="audio/"):
        self.tts.tts_to_file(text=text, file_path=f"{path}basic_tts_{self.saving_count}.wav")

    @staticmethod
    def speak_from_path(path):
        audio = AudioSegment.from_file(path, format="wav")
        play(audio)
