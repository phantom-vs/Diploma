"""Предобработка EEG: загрузка, реестр постпроцессоров."""

import preproc.postprocess
from preproc.registry import REGISTRY, register_postprocessor

__all__ = ["REGISTRY", "register_postprocessor"]
