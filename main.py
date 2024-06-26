from fastapi import FastAPI
import whisper
import numpy as np
from pydub import AudioSegment
from voice_sample import VoiceSample

app = FastAPI()


class Transcribe:
    _model: whisper.model

    def __init__(self):
        self._model = whisper.load_model("medium")

    def get_transcript(self, sound: VoiceSample) -> str:
        return self._model.transcribe(sound.get_sample_as_np_array(), language='pl')['text'].strip()


@app.get("/request/")
async def request(audio: VoiceSample):
    global transcribe
    out = transcribe.get_transcript(audio)
    print(out)
    return out


@app.get("/")
async def root():
    return {"message": "Hello World"}


def init():
    global transcribe
    transcribe = Transcribe()


init()
