import os
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Audio:
	def __init__(self, speaker, text, audio):
		self.speaker = speaker
		self.text = text
		self.audio = audio

	def __str__(self):
		return f'{self.speaker}, {self.text}'

def loadnpy(filepath, word='*', verbose=1):
	filepath = filepath.strip('/')
	x = []

	for word_path in sorted(glob.glob(f'{filepath}/{word}/*')):
		speaker = int(word_path.split('/')[-1].split('-')[0])
		text = word_path.split('/')[-2]
		a = np.load(word_path).astype(np.float32)
		audio = Audio(speaker, text, a)
		x.append(audio)

		if verbose == 1:
			print(audio)
	
	return x

def padding(audio_list):
	max_len = 0
	for audio in audio_list:
		if audio.audio.shape[0] > max_len:
			max_len = audio.audio.shape[0]

	for i in range(len(audio_list)):
		audio_list[i].audio = np.concatenate([np.zeros([max_len - audio_list[i].audio.shape[0], audio_list[i].audio.shape[1]]).astype(np.float32), audio_list[i].audio], 0)

	return audio_list

def make_list(audio_list):
	speaker_list = dict()

	for audio in audio_list:
		if audio.speaker not in speaker_list:
			speaker_list[audio.speaker] = [audio]
		else:
			speaker_list[audio.speaker].append(audio)

	return speaker_list

def normalize(audio_list, mean=None, std=None):
	if mean is None:
		mean, l = 0, 0
		for audio in audio_list:
			l += audio.audio.shape[0] * audio.audio.shape[1]
			mean += np.sum(audio.audio)
		mean /= l

	if std is None:
		std, l = 0, 0
		for audio in audio_list:
			l += audio.audio.shape[0] * audio.audio.shape[1]
			std += np.sum((audio.audio - mean) ** 2)
		std /= l
		std = np.sqrt(std)
		
	for i in range(len(audio_list)):
		audio_list[i].audio = (audio_list[i].audio - mean) / std

	return audio_list, mean, std
