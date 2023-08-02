#!/usr/bin/env python

from misc.config import MODEL_NAME, MODEL_DIR
from flagai.auto_model.auto_loader import AutoLoader

loader = AutoLoader(task_name="txt_img_matching",
                    model_name=MODEL_NAME,
                    model_dir=MODEL_DIR)

loader.get_model()
