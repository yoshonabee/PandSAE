# convert librispeech to montreal-force-aligner format
import glob
import sys
import os
from pydub import AudioSegment

if len(sys.argv) < 3:
	print('usage: python3 libri2mfa.py [original libispeech directory(ex. dev-clean)] [mfa format directory]')
	exit(0)

root_dir = sys.argv[1]
mfa_dir = sys.argv[2]

dset = root_dir.strip('/').split('/')[-1]

for utt_dir in sorted(glob.glob(f'{root_dir}/*/*')):
	speaker_id = utt_dir.strip('/').split('/')[-2]
	utt_id = utt_dir.strip('/').split('/')[-1]
	# flac -> wav, sr=16000
	dest_dir = f'{mfa_dir}/{dset}/{speaker_id}'

	if not os.path.isdir(dest_dir):
		os.makedirs(dest_dir)

	for flac_path in sorted(glob.glob(f'{utt_dir}/*.flac')):
		flac_filename = flac_path.split('/')[-1][:-5]
		audio = AudioSegment.from_file(flac_path, format='flac', frame_rate=16000)
		audio.export(f'{dest_dir}/{flac_filename}.wav', format='wav')

	# transcription -> .lab
	with open(f'{utt_dir}/{speaker_id}-{utt_id}.trans.txt', 'r') as f_in:
		for line in f_in:
			 id, text = line.strip().split(' ', maxsplit=1)
			 with open(f'{dest_dir}/{id}.lab', 'w') as f_out:
				 f_out.write(f'{text}\n')