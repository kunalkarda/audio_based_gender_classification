from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import uvicorn
from io import BytesIO

app = FastAPI()

MODEL_NAME = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/predict-gender/")
async def predict_gender(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_data, sample_rate = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
        inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        gender = "Male" if pred_label == 1 else "Female"
        confidence = probs.max().item()
        return {"gender": gender, "confidence": f"{confidence:.2%}"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
