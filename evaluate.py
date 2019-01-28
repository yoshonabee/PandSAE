from model import *
from utils import *

import sys
import numpy as np
import torch

if len(sys.argv) < 4:
	print('usage: python3 evaluate.py [testing data directory] [word] [cuda]')
	exit(0)

if sys.argv[2] != '-1':
	os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
	cuda = True

mean = np.load("ph_mean.npy")
std = np.load('ph_std.npy')


filepath = sys.argv[1]

word = loadnpy(filepath, sys.argv[2])
word, _, _ = normalize(word, mean, std)
# word = padding(word)
print(len(word))

word2 = loadnpy(filepath, 'mother')

Ep = Encoder()
Es = Encoder()

Ep.load_state_dict(torch.load("Ep.pkl", map_location='cpu'))
Es.load_state_dict(torch.load("Es.pkl", map_location='cpu'))

a = torch.from_numpy(word[0].audio).float().unsqueeze(0)
b = torch.from_numpy(word[15].audio).float().unsqueeze(0)
c = torch.from_numpy(word2[0].audio).float().unsqueeze(0)

print(word[14].speaker, word[15].speaker)
a = Ep(a)
b = Ep(b)
c = Ep(c)

print(a - b)