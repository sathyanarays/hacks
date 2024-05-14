import argparse

from torchvision import models
from typing import Dict, Union
import torch
import numpy as np
import kserve
from kserve import Model, ModelServer
from transformers import AutoProcessor, AutoModelForVision2Seq
import base64
from PIL import Image
import io
import uuid

class AlexNetModel(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b-base",
        ).to("cpu")
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")        
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        img_data = payload["instances"][0]["image"]["b64"]
        raw_img_data = base64.b64decode(img_data)
        input_image = Image.open(io.BytesIO(raw_img_data))

        input_text = payload["instances"][0]["text"]

        prompts = [input_text]  
        images = [[input_image]]
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}


# Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)        

        result = generated_texts
        response_id = str(uuid.uuid4())
        return {"predictions": result}

if __name__ == "__main__":
    model = AlexNetModel("custom-model")
    ModelServer().start([model])
