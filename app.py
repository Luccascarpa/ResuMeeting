from flask import Flask, redirect, render_template, request

import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Suggestion: https://github.com/openai/whisper/discussions/264 runnnig the pyannote.audio first and then just running whisper on the split-by-speaker chunks

# Aux funcs
def segment_embedding(segment, audio, path, duration):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def time(secs):
    return datetime.timedelta(seconds=round(secs))

app = Flask(__name__)

@app.get('/')
def index():
    return render_template('index.html')

@app.post('/transcript')
def transcript():

    number_of_speakers = request.values.get('number of speakers')
    language = request.values.get('language')

    model_size = request.values.get('model size') # ['tiny', 'base', 'small', 'medium', 'large']

    file = request.files['file']
    path = os.path.join('./storage/audios', file.filename)

    if file.filename[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
        path = os.path.join('./storage/audios', f'{file.filename[:-4]}.wav')
    
    file.save(path)

    model = whisper.load_model(model_size)
    results = model.transcribe(path)
    
    segments = results["segments"]
    transcription = results["text"]

    with contextlib.closing(wave.open(path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()

    embbedings = np.zeros(shape=(len(segments),192))
    for i , segment in enumerate(segments):
        embbedings[i] = segment_embedding(segment)
    embbedings = np.nan_to_num(embbedings)

    clustering = AgglomerativeClustering(number_of_speakers).fit(embbedings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

    f = open(os.path.join('./storage/audios', f'{file.filename[:-4].txt}'), 'w')

    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i -1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + "\n")
        f.write(segment["text"][1:] + ' ')
    f.close()

    transcription_with_speakers = open(os.path.join('./storage/audios', f'{file.filename[:-4].txt}'), 'r').read()

    print(transcription_with_speakers)

    return transcription_with_speakers

if __name__ == '__main__':
    app.run()