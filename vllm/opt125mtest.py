from transformers import pipeline
generator = pipeline('text-generation', model="facebook/opt-125m", device="cuda")
out = generator("What are we having for dinner?")
import time
time.sleep(100)
print(out)