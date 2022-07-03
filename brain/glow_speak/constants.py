#!/usr/bin/env python3

"""Constants and enums"""
from enum import Enum

"""Glow-speak web server"""

import asyncio
import typing
from dataclasses import dataclass
import logging
import numpy as np
import onnxruntime

from espeak_phonemizer import Phonemizer
from pathlib import Path
PAD = "_"

_MISSING = object()

_DIR = Path(__file__).parent
_TEMPLATE_DIR = _DIR / "templates"
_TEMP_DIR: typing.Optional[str] = None

_LOGGER = logging.getLogger("glow_speak.http_server")
_LOOP = asyncio.get_event_loop()

class VocoderQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_SESSION_OPTIONS = onnxruntime.SessionOptions()


_TTS_INFO_LOCK = asyncio.Lock()


@dataclass
class VocoderInfo:
    model: typing.Any
    sample_rate: int
    channels: int
    sample_bytes: int
    bias_spec: typing.Optional[np.ndarray] = None


# quality -> onnx model
_VOCODER_INFO: typing.Dict[VocoderQuality, VocoderInfo] = {}
_VOCODER_INFO_LOCK = asyncio.Lock()


# language -> phonemizer
_PHONEMIZERS: typing.Dict[str, Phonemizer] = {}
_PHONEMIZERS_LOCK = asyncio.Lock()


@dataclass(frozen=True)  # must be hashable
class TextToWavParams:
    text: str
    voice: str
    text_language: str
    vocoder_quality: VocoderQuality
    denoiser_strength: float
    noise_scale: float
    length_scale: float


_WAV_CACHE: typing.Dict[TextToWavParams, Path] = {}