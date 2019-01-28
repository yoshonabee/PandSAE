from utils import *
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.GRU1 = nn.GRU(39, 128, batch_first=True)
		self.GRU2 = nn.GRU(128, 128, batch_first=True)

	def forward(self, x):
		x, h1 = self.GRU1(x)
		_, h2 = self.GRU2(x)
		return h2

class Decoder(nn.Module):
	def __init__(self, length, cuda):
		super(Decoder, self).__init__()
		self.length = length
		self.GRU1 = nn.GRU(256, 256, batch_first=True)
		self.GRU2 = nn.GRU(256, 256, batch_first=True)
		self.linear = nn.Linear(256, 39)
		self.c = cuda

	def forward(self, pho, spe):
		z = torch.zeros([pho.shape[1], self.length, pho.shape[2] + spe.shape[2]], dtype=torch.float32)

		if self.c:
			z = z.cuda()

		h = torch.cat([pho, spe], 2)
		
		x, _ = self.GRU1(z, h)
		x, _ = self.GRU2(x)
		
		x = x.contiguous().view(-1, x.shape[2])
		x = self.linear(x)
		x = x.view(-1, self.length, 39)

		return(x)

class ReconstructLoss(nn.Module):
	def __init__(self):
		super(ReconstructLoss, self).__init__()

	def forward(self, recon, target):
		dis = torch.sum((recon - target) ** 2, 1)
		return torch.mean(dis / recon.size(0))

class SpeakerLoss(nn.Module):
	def __init__(self, l):
		super(SpeakerLoss, self).__init__()
		self.l = l

	def forward(self, s, vs, other_s, other_vs):
		d = torch.sum((vs - other_vs) ** 2, 1)
		
		w = torch.eq(s, other_s).float()
		for i in range(w.shape[0]):
			if w[i] == 0: d[i] = max(self.l - d[i], 0)

		return torch.mean(d)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.l1 = nn.Linear(256, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, 1)

	def forward(self, vp):
		x = self.l1(vp)
		x = self.l2(x)
		x = self.l3(x)

		return x

class DiscriminatorLoss(nn.Module):
	def __init__(self):
		super(DiscriminatorLoss, self).__init__()

	def forward(self, s, other_s, x):
		w = torch.eq(s, other_s).float()
		for i in range(w.shape[0]):
			if w[i] == 0: w[i] = -1
		
		return torch.mean(w.mul(x))

class training_set(Dataset):
	def __init__(self, audio_list, speaker_list):
		self.audio_list = audio_list
		self.speaker_list = speaker_list

	def __getitem__(self, index):
		s = self.audio_list[index].speaker
		audio = self.audio_list[index].audio

		if random.random() > 0.5:
			speaker_audio = random.choice(self.speaker_list[s])
		else:
			speaker_audio = random.choice(self.audio_list)

		# speaker_audio = random.choice(self.audio_list)

		speaker_s = speaker_audio.speaker
		speaker_audio = speaker_audio.audio

		if random.random() > 0.5:
			other_audio = random.choice(self.speaker_list[s])
		else:
			other_audio = random.choice(self.audio_list)
		
		# other_audio = random.choice(self.audio_list)
		
		other_s = other_audio.speaker
		other_audio = other_audio.audio

		positive_audio = random.choice(self.speaker_list[s])
		negative_audio = random.choice(self.audio_list)

		while negative_audio.speaker == s:
			negative_audio = random.choice(self.audio_list)

		return (s, audio), (speaker_s, speaker_audio), (other_s, other_audio), (positive_audio.audio, negative_audio.audio)

	def __len__(self):
		return len(self.audio_list)