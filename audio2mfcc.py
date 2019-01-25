import glob
import sys
import os
import librosa

import numpy as np

if len(sys.argv) < 3:
	print('usage: python3 audio2mfcc.py [audio directory(ex. dev-clean)] [output npy directory]')
	exit(0)

root_dir = sys.argv[1].strip('/')
output_dir = sys.argv[2].strip('/')

dset = root_dir.strip('/').split('/')[-1]

for word_dir in sorted(glob.glob(f'{root_dir}/*')):
	word_id = word_dir.split('/')[-1]
	dest_dir = f'{output_dir}/{word_id}'
	
	if not os.path.isdir(dest_dir):
		os.makedirs(dest_dir)

	for wav_path in sorted(glob.glob(f'{word_dir}/*.wav')):
		wav_name = wav_path.split('/')[-1][:-4]

		y, sr = librosa.load(wav_path, sr=16000)
		y = librosa.feature.mfcc(y, sr, n_mfcc=39).T

		npy_name = f'{dest_dir}/{wav_name}.npy'
		np.save(npy_name, y)
		print(npy_name)

