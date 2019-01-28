from utils import *
from model import *

import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import grad

cuda = False

if len(sys.argv) < 3:
	print('usage: python3 train.py [training data directory] [model path] [cuda]')
	exit(0)

if sys.argv[3] != '-1':
	os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
	cuda = True

filepath = sys.argv[1]
modelpath = sys.argv[2].strip('/')

LR = 0.0001
BATCH_SIZE = 32
EPOCH = 500
LAMBDA = 10

audio_list = loadnpy(sys.argv[1], verbose=1)
audio_list, mean, std = normalize(audio_list)
np.save('ph_mean.npy', mean)
np.save('ph_std.npy', std)

audio_list = padding(audio_list)
speaker_list = make_list(audio_list)

train_set = training_set(audio_list, speaker_list)
dataLoader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

del audio_list, speaker_list

Ep = Encoder()
Es = Encoder()
Dec = Decoder(train_set.audio_list[0].audio.shape[0], cuda)
ReconCriterion = ReconstructLoss()
Dis = Discriminator()
DisLoss = DiscriminatorLoss()
SpeakerCriterion = SpeakerLoss(0.01)

EpOptim = Adam(Ep.parameters(), lr=LR, betas=(0.5, 0.9))
EsOptim = Adam(Es.parameters(), lr=LR, betas=(0.5, 0.9))
DecOptim = Adam(Dec.parameters(), lr=LR, betas=(0.5, 0.9))
DisOptim = Adam(Dis.parameters(), lr=LR, betas=(0.5, 0.9))

if cuda:
	Ep.cuda()
	Es.cuda()
	Dec.cuda()
	ReconCriterion.cuda()
	Dis.cuda()
	SpeakerCriterion.cuda()
	DisLoss.cuda()

for epoch in range(EPOCH):
	if (epoch + 1) % 4 == 0:
		for i, ((s, audio), (speaker_s, speaker_audio), (other_s, other_audio), (positive_audio, negative_audio)) in enumerate(dataLoader):
			EpOptim.zero_grad()
			EsOptim.zero_grad()
			DecOptim.zero_grad()
			DisOptim.zero_grad()

			if cuda:
				s = s.cuda()
				audio = audio.cuda()
				speaker_s = speaker_s.cuda()
				speaker_audio = speaker_audio.cuda()
				other_s = other_s.cuda()
				other_audio = other_audio.cuda()
				positive_audio = positive_audio.cuda()
				negative_audio = negative_audio.cuda()
			
			pho = Ep(audio)
			spe = Es(audio)
			recon = Dec(pho, spe)
			reconstructLoss = ReconCriterion(recon, audio)

			same_spe = Es(speaker_audio)
			spe = spe.view(spe.size(1), -1)
			same_spe = same_spe.view(same_spe.size(1), -1)
			speaker_loss = SpeakerCriterion(s, spe, speaker_s, same_spe)

			other = Ep(other_audio)
			pho = pho.view(pho.size(1), -1)
			other = other.view(other.size(1), -1)
			vp = torch.cat([pho, other], 1)
			dis_value = Dis(vp)
			dis_loss = DisLoss(s, other_s, dis_value)

			# positive = Ep(positive_audio)
			# negative = Ep(negative_audio)
			# positive = positive.view(positive.size(1), -1)
			# negative = negative.view(negative.size(1), -1)
			# positive = torch.cat([pho, positive], 1)
			# negative = torch.cat([pho, negative], 1)

			# alpha = torch.rand(positive.size(0), 1).cuda()
			# alpha = alpha.expand_as(positive)

			# vp = alpha * positive + (1 - alpha) * negative
			# dis_value = Dis(vp)

			# dis_grad = grad(dis_value, vp, grad_outputs=torch.ones(dis_value.size()).cuda() if cuda else torch.ones(vp.size()), retain_graph=True, create_graph=True)[0]
			
			# dis_reg = torch.mean(torch.sqrt(torch.sum(dis_grad ** 2, 1) + 1e-12))

			loss = reconstructLoss + speaker_loss + dis_loss
			loss.backward()

			EpOptim.step()
			EsOptim.step()
			DecOptim.step()

			if (i + 1) % 100 == 0:
				print(f'epoch:{epoch + 1} | iter:{i + 1} | loss:{loss.item()} | reg:{dis_reg.item()}')
	else:
		for i, ((s, audio), _, (other_s, other_audio), (positive_audio, negative_audio)) in enumerate(dataLoader):
			EpOptim.zero_grad()
			EsOptim.zero_grad()
			DecOptim.zero_grad()
			DisOptim.zero_grad()

			if cuda:
				s = s.cuda()
				audio = audio.cuda()
				other_s = other_s.cuda()
				other_audio = other_audio.cuda()
				positive_audio = positive_audio.cuda()
				negative_audio = negative_audio.cuda()

			pho = Ep(audio)
			other = Ep(other_audio)
			pho = pho.view(pho.size(1), -1)
			other = other.view(other.size(1), -1)
			vp = torch.cat([pho, other], 1)
			dis_value = Dis(vp)
			dis_loss = DisLoss(s, other_s, dis_value)

			positive = Ep(positive_audio)
			negative = Ep(negative_audio)
			positive = positive.view(positive.size(1), -1)
			negative = negative.view(negative.size(1), -1)
			positive = torch.cat([pho, positive], 1)
			negative = torch.cat([pho, negative], 1)

			alpha = torch.rand(positive.size(0), 1).cuda()
			alpha = alpha.expand_as(positive)

			vp = alpha * positive + (1 - alpha) * negative
			dis_value = Dis(vp)

			dis_grad = grad(dis_value, vp, grad_outputs=torch.ones(dis_value.size()).cuda() if cuda else torch.ones(vp.size()), retain_graph=True, create_graph=True)[0]
			
			dis_reg = torch.mean(torch.sqrt(torch.sum(dis_grad ** 2, 1) + 1e-12))

			loss = -dis_loss + LAMBDA * dis_reg
			loss.backward()

			DisOptim.step()

			if (i + 1) % 100 == 0: print(f'epoch:{epoch + 1} | iter:{i + 1} | loss:{dis_loss.item()} | reg:{dis_reg.item()} | D')
	
torch.save(Ep.state_dict(), "Ep.pkl")
torch.save(Es.state_dict(), "Es.pkl")
torch.save(Dec.state_dict(), "Des.pkl")
torch.save(Dis.state_dict(), "Dis.pkl")