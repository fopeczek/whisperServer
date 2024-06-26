from __future__ import annotations

import base64
import wave
from pathlib import Path

import numpy as np
import pyaudio
from pydantic import BaseModel, field_serializer, field_validator
from pydub import AudioSegment


class VoiceSample(BaseModel):
    data: bytes  # Annotated[bytes, BeforeValidator(VoiceSample.deserialize_data)]
    frame_rate: int
    sample_width: int = 2

    @field_validator('data', mode='before')
    @classmethod
    def validate_data(cls, data: bytes) -> bytes:
        if isinstance(data, str):
            return base64.b85decode(data)
        else:
            assert isinstance(data, bytes)
            return data

    @field_serializer('data')
    def serialize_data(self, data: bytes, _info):
        return base64.b85encode(data)

    def get_sample_as_np_array(self) -> np.ndarray:
        audio_segment = AudioSegment(
            self.data,
            frame_rate=self.frame_rate,
            sample_width=self.sample_width,
            channels=1
        )

        if self.frame_rate != 16000:  # 16 kHz
            audio_segment = audio_segment.set_frame_rate(16000)
        arr = np.array(audio_segment.get_array_of_samples())
        arr = arr.astype(np.float32) / 32768.0
        return arr

    def ResampledClone(self, frame_rate: int = 16000) -> VoiceSample:
        audio_segment = AudioSegment(
            self.data,
            frame_rate=self.frame_rate,
            sample_width=self.sample_width,
            channels=1
        )

        audio_segment = audio_segment.set_frame_rate(frame_rate)
        return VoiceSample(data=audio_segment.raw_data, frame_rate=frame_rate, sample_width=self.sample_width)

    def save(self, filename: Path):
        # Save the last recording as WAV file
        wf = wave.open(str(filename), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.frame_rate)
        wf.writeframes(self.data)
        wf.close()

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self.data

    def play(self):
        # Play the last recording
        p = pyaudio.PyAudio()
        if self.sample_width == 2:
            p_format = pyaudio.paInt16
        else:
            raise ValueError("Unsupported sample width")
        stream = p.open(format=p_format,
                        channels=1,
                        rate=self.frame_rate,
                        output=True)
        stream.write(self.data)
        stream.stop_stream()

    def length(self):
        return len(self.data) / self.frame_rate / self.sample_width


def test_voice_sample() -> VoiceSample:
    data = "ala ma kota".encode()
    print(data)
    frame_rate = 44100
    print(base64.b85encode(data))
    sample_width = 2
    voice_sample = VoiceSample(data=data, frame_rate=frame_rate, sample_width=sample_width)
    print(str(voice_sample))
    print(voice_sample.model_dump_json())
    return voice_sample


if __name__ == '__main__':
    json_txt = test_voice_sample().json()
    with open("test.json", "w") as f:
        f.write(json_txt)

    voice_sample = VoiceSample.parse_raw(json_txt)
    print(f"voice_sample: {voice_sample}")
