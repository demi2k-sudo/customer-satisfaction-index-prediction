from gpt import LLM
from speech_brain_app import get_emotion
from transcription import Transcriptor


class csi:

    def __init__(self,api_base, api_version, api_key):
        self.gpt = LLM(api_base, api_version, api_key)
        self.transcriptor = Transcriptor()

    def process(self, path):
        emotions = get_emotion(path)
        transcripts = self.transcriptor.transcribe(path)
        result = self.gpt.get_csi(transcripts, emotions)
        result = result["choices"][0]["message"]["content"]
        return result
