import tensorflow as tf
import tensorflow_hub as hub
import csv
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

__MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
__SAMPLE_RATE = 16000
__CLASS_NAMES = []
with tf.io.gfile.GFile(__MODEL.class_map_path().numpy()) as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    __CLASS_NAMES.append(row['display_name'])

def classify(file: str):
  # Load file
  if not os.path.exists(file):
    raise FileNotFoundError("File not found")
  sample_rate, wav_data = wavfile.read(file, 'rb')
  
  # Stereo check
  if len(wav_data.shape) > 1:
    wav_data = __to_mono(wav_data)

  # Resample check
  if sample_rate != __SAMPLE_RATE:
    wav_data = __resample(wav_data, sample_rate)
    
  # Normalize
  wav_data = __normalize(wav_data)

  scores, embeddings, spectrogram = __MODEL(wav_data)
  scores_np = scores.numpy()
  infered_class = __CLASS_NAMES[scores_np.mean(axis=0).argmax()]
  return infered_class

def __to_mono(data):
  return data.sum(axis=1) / 2

def __normalize(data):
  return  data / tf.int16.max

def __resample(data, sample_rate: int):
  desired_length = int(round(float(len(data)) / sample_rate * __SAMPLE_RATE))
  new_data = signal.resample(data, desired_length)
  return new_data