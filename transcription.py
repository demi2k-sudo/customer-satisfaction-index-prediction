import torch
from transformers import pipeline
from datasets import load_dataset

class Transcriptor:
  def __init__(self):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.pipe = pipeline(
      "automatic-speech-recognition",
      model="openai/whisper-base",
      chunk_length_s=30,
      device=device,
    )
  
  def transcribe(self,path): 
    prediction = self.pipe(path, batch_size=8, return_timestamps=True)["chunks"]
    print(prediction)

    return prediction
    
    




