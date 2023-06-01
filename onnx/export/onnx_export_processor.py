#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from misc.config import MODEL_FP, PROCESS_DIR
from os import makedirs

makedirs(PROCESS_DIR, exist_ok=True)
proc = AltCLIPProcess.from_pretrained(MODEL_FP)
proc.save_pretrained(PROCESS_DIR)
