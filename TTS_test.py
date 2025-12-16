import torch
from TTS.utils.radam import RAdam
import torch.serialization
from TTS.api import TTS

from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd

# model_name = "tts_models/es/mai/tacotron2-DDC"
# model_name = "tts_models/en/ljspeech/glow-tts"
model_name = "tts_models/es/css10/vits"
# model_name = "tts_models/multilingual/multi-dataset/your_tts"


class Speaker:
    def __init__(self, device= "cuda" if torch.cuda.is_available() else "cpu"):
        torch.serialization.add_safe_globals([RAdam])
        self.tts = TTS(model_name).to(device)
        self.saving_count = 0

    def speak(self, text):
        wav = self.tts.tts(text)
        sd.play(wav, samplerate=22050)
        sd.wait()

    def save_dialog(self, text, path="audio/"):
        self.tts.tts_to_file(text=text, file_path=f"{path}basic_tts_{self.saving_count}.wav")

    @staticmethod
    def speak_from_path(path):
        audio = AudioSegment.from_file(path, format="wav")
        play(audio)
