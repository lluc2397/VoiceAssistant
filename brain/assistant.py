#!/usr/bin/env python3

import asyncio
from dataclasses import dataclass
from espeak_phonemizer import Phonemizer
import functools
from glow_speak import (
    PhonemeGuesser,
    VocoderQuality,
    get_vocoder_dir,
    ids_to_mels,
    init_denoiser,
    mels_to_audio,
    text_to_ids,
)
from glow_speak.download import find_voice
import hashlib
import io
import json
import onnxruntime
import os
from pathlib import Path
from phonemes2ids import load_phoneme_ids, load_phoneme_map
import queue
import sounddevice as sd
import soundfile as sf
import string
import time
import typing
from vosk.vosk import Model, KaldiRecognizer
from glow_speak.constants import (
    _TTS_INFO_LOCK, _VOCODER_INFO, VocoderQuality, _LOGGER, _LOOP,
    _SESSION_OPTIONS, TextToWavParams, _TEMP_DIR, _WAV_CACHE, _PHONEMIZERS_LOCK,
    _VOCODER_INFO_LOCK, VocoderInfo, _PHONEMIZERS
)
import wave


@dataclass
class TTSInfo:
    model: typing.Any
    phoneme_to_id: typing.Dict[str, int]
    phoneme_guesser: PhonemeGuesser
    language: typing.Optional[str] = None
    phoneme_map: typing.Optional[typing.Dict[str, typing.List[str]]] = None

_TTS_INFO: typing.Dict[str, TTSInfo] = {}

class Listen:
    queue = queue.Queue()
    model = Model(f"{Path(__file__).resolve(strict=True).parent.parent.parent}/vosk-model-small-es-0.22")
    default_device = None
    device_info = sd.query_devices(default_device, 'input')
    default_samplerate = int(device_info['default_samplerate'])
        

    def callback(self, indata, frames, time, status):
        self.queue.put(bytes(indata))
    
    def listen(self):
        try:    
            with sd.RawInputStream(
                samplerate=self.default_samplerate, 
                blocksize = 8000, 
                device=self.default_device, 
                dtype='int16',
                channels=1, callback=self.callback
                ):
                    rec = KaldiRecognizer(self.model, self.default_samplerate)
                    while True:
                        data = self.queue.get()
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            print(result) # Actual resutl with conf stat
                            if 'text' in result:
                                voice_command = result['text']
                            else:
                                voice_command = result['word']
                            return voice_command

        except KeyboardInterrupt:
            print('\nDone')
        except Exception as e:
            print(e)


class Talk:
    async def text_to_wav(
        self,
        text: str,
        voice: str,
        text_language: typing.Optional[str],
        vocoder_quality: VocoderQuality,
        denoiser_strength: float,
        noise_scale: float,
        length_scale: float,
        no_cache: bool,
    ) -> bytes:
        """Runs TTS for each line and accumulates all audio into a single WAV."""
        # Load TTS model from cache or disk
        async with _TTS_INFO_LOCK:
            maybe_tts = _TTS_INFO.get(voice)
            if maybe_tts is None:
                voice_dir = find_voice(voice, voices_dir=f'{Path(__file__).parent.parent}/in-.local-share/glow-speak/voices')
                assert voice_dir is not None, f"Voice not found: {voice}"

                tts_model = await _LOOP.run_in_executor(
                    None,
                    functools.partial(
                        onnxruntime.InferenceSession,
                        str(voice_dir / "generator.onnx"),
                        sess_options=_SESSION_OPTIONS,
                    ),
                )

                # Load model language
                model_language: typing.Optional[str] = None
                lang_path = voice_dir / "LANGUAGE"
                if lang_path.is_file():
                    model_language = lang_path.read_text().strip()

                # Load phoneme -> id map
                with open(
                    voice_dir / "phonemes.txt", "r", encoding="utf-8"
                ) as phonemes_file:
                    phoneme_to_id = load_phoneme_ids(phonemes_file)

                # Load phoneme -> phoneme map
                phoneme_map = None
                phoneme_map_path = voice_dir / "phoneme_map.txt"
                if phoneme_map_path.is_file():
                    with open(phoneme_map_path, "r", encoding="utf-8") as phoneme_map_file:
                        phoneme_map = load_phoneme_map(phoneme_map_file)

                phoneme_guesser = PhonemeGuesser(phoneme_to_id, phoneme_map)

                # Add to cache
                tts_info = TTSInfo(
                    model=tts_model,
                    phoneme_to_id=phoneme_to_id,
                    phoneme_map=phoneme_map,
                    phoneme_guesser=phoneme_guesser,
                    language=model_language,
                )

                _TTS_INFO[voice] = tts_info
            else:
                tts_info = maybe_tts

        if text_language is None:
            text_language = tts_info.language

        assert (
            text_language is not None
        ), "Text language not set (missing LANGUAGE file in voice directory)"

        text_to_wav_params: typing.Optional[TextToWavParams] = None
        if _TEMP_DIR and (not no_cache):
            # Look up in cache
            text_to_wav_params = TextToWavParams(
                text=text,
                voice=voice,
                text_language=text_language,
                vocoder_quality=vocoder_quality,
                noise_scale=noise_scale,
                length_scale=length_scale,
                denoiser_strength=denoiser_strength,
            )

            maybe_wav_path = _WAV_CACHE.get(text_to_wav_params)
            if (maybe_wav_path is not None) and maybe_wav_path.is_file():
                wav_bytes = maybe_wav_path.read_bytes()
                return wav_bytes

        # Load language info
        async with _PHONEMIZERS_LOCK:
            maybe_phonemizer = _PHONEMIZERS.get(text_language)
            if maybe_phonemizer is None:
                # Initialize eSpeak phonemizer
                phonemizer = Phonemizer(default_voice=text_language)
            else:
                phonemizer = maybe_phonemizer

        # Load vocoder from cache or disk
        async with _VOCODER_INFO_LOCK:
            maybe_vocoder = _VOCODER_INFO.get(vocoder_quality)
            if maybe_vocoder is None:
                vocoder_dir = get_vocoder_dir(vocoder_quality)

                vocoder_model = await _LOOP.run_in_executor(
                    None,
                    functools.partial(
                        onnxruntime.InferenceSession,
                        str(vocoder_dir / "generator.onnx"),
                        sess_options=_SESSION_OPTIONS,
                    ),
                )

                bias_spec = None

                # Load audio config
                with open(
                    vocoder_dir / "config.json", "r", encoding="utf-8"
                ) as vocoder_config_file:
                    vocoder_config = json.load(vocoder_config_file)
                    vocoder_audio = vocoder_config["audio"]
                    num_mels = int(vocoder_audio["num_mels"])
                    sample_rate = int(vocoder_audio["sampling_rate"])
                    channels = int(vocoder_audio["channels"])
                    sample_bytes = int(vocoder_audio["sample_bytes"])

                    if denoiser_strength > 0:
                        bias_spec = await _LOOP.run_in_executor(
                            None, functools.partial(init_denoiser, vocoder_model, num_mels)
                        )

                    vocoder_info = VocoderInfo(
                        model=vocoder_model,
                        sample_rate=sample_rate,
                        channels=channels,
                        sample_bytes=sample_bytes,
                        bias_spec=bias_spec,
                    )

                    _VOCODER_INFO[vocoder_quality] = vocoder_info
            else:
                vocoder_info = maybe_vocoder

        # Synthesize each line separately.
        # Accumulate into a single WAV file.
        audios = []

        with io.StringIO(text) as lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                text_ids = text_to_ids(
                    text=line,
                    phonemizer=phonemizer,
                    phoneme_to_id=tts_info.phoneme_to_id,
                    phoneme_map=tts_info.phoneme_map,
                    missing_func=tts_info.phoneme_guesser.guess_ids,
                )
                mels = ids_to_mels(
                    ids=text_ids,
                    tts_model=tts_info.model,
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                )
                audio = mels_to_audio(
                    mels=mels,
                    vocoder_model=vocoder_info.model,
                    denoiser_strength=denoiser_strength,
                    bias_spec=vocoder_info.bias_spec,
                )
                audios.append(audio)

        with io.BytesIO() as wav_io:
            wav_out: wave.Wave_write = wave.open(wav_io, "wb")
            with wav_out:
                wav_out.setframerate(vocoder_info.sample_rate)
                wav_out.setnchannels(vocoder_info.channels)
                wav_out.setsampwidth(vocoder_info.sample_bytes)
                wav_out.writeframes(audio.tobytes())

            wav_bytes = wav_io.getvalue()

        if _TEMP_DIR and (text_to_wav_params is not None) and (not no_cache):
            try:
                # Save to cache
                text_filtered = text.strip().replace(" ", "_")
                text_filtered = text_filtered.translate(
                    str.maketrans("", "", string.punctuation.replace("_", ""))
                )

                param_hash = hashlib.md5()
                param_hash.update(str(text_to_wav_params).encode("utf-8"))

                output_name = "{text:.100s}_{hash}.wav".format(
                    text=text_filtered, hash=param_hash.hexdigest()
                )
                output_path = os.path.join(_TEMP_DIR, output_name)

                with open(output_path, mode="wb") as output_file:
                    output_file.write(wav_bytes)
                    _WAV_CACHE[text_to_wav_params] = Path(output_path)

            except Exception as e:
                _LOGGER.exception(f"text_to_wav: {e}")

        return wav_bytes

    
    async def wav_to_voice(self, text: str):

        # TTS settings
        tts_args: typing.Dict[str, typing.Any] = {
            'noise_scale': float(0.333),
            'length_scale': float(1.0),
            'vocoder_quality': VocoderQuality.HIGH,
            'denoiser_strength': float(0.005),
            'text_language': None,
            'no_cache': False,
            'voice': 'es_tux'
        }

        wav_bytes = await self.text_to_wav(text, **tts_args)

        with open('myfile.wav', mode='bx') as f:
            f.write(wav_bytes)

        filename = 'myfile.wav'
        # Extract data and sampling rate from file
        data, fs = sf.read(filename, dtype='float32')  
        sd.play(data, fs)
        status = sd.wait()


class Assistant:
    stt = Listen
    tts = Talk
    wake_words = ['alexa', 'espejito']
    actions = ['habla', 'canta', '']

    def retreive_actions(self, voice_command: str):
        if voice_command in self.actions:
            return voice_command

    def do_action(self, action: str):
        if action == 'canta':
            return self.speak('na na na na nananana nanana nanana nanana nanana na')

    async def wait_action(self):
        while True:
            listener_response = self.stt.listen()
            action = self.retreive_action(listener_response)
            if action:
                await self.speak('Voooooy')
                self.do_action(action)
                return self.wait_wake_word()
            else:
                await self.speak('No se hacer eso. En qué más puedo ayudarte')
                return self.wait_action()

    def wait_wake_word(self):
        while True:
            listener_response = self.stt.listen()
            if listener_response in self.wake_words:
                return self.wait_action()
    
    async def speak(self, text: str):
        return await self.tts.wav_to_voice(text)

