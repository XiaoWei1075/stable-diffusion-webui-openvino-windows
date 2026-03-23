# Copyright (C) 2024-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class ModelState:
    def __init__(self):
        self.enable_ov_extension = True
        self.enable_caching = True
        self.recompile = True
        self.device = "GPU"
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.mode = 0
        self.partition_id = 0
        self.model_name = ""
        self.control_models = []
        self.is_sdxl = False
        self.lora_model = "None"
        self.vae_ckpt = "None"
        self.refiner_ckpt = "None"


model_state = ModelState()

pipes = {'diffusers': None, 'openvino': None}
