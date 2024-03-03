from speechbrain.inference.diarization import Speech_Emotion_Diarization

def get_emotion(path):
    classifier = Speech_Emotion_Diarization.from_hparams(source="speechbrain/emotion-diarization-wavlm-large",run_opts={"device":"cuda"})
    diary = classifier.diarize_file(path)
    return diary

path = r"new.wav"
print(get_emotion(path))