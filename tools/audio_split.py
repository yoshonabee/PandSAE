import glob
import sys
import os
from pydub import AudioSegment
import tgt

if len(sys.argv) < 4:
	print('usage: python3 audio_split.py [original libispeech directory(ex. dev-clean)] [mfa result directory] [output directory]')
	exit(0)

root_dir = sys.argv[1]
tg_dir = sys.argv[2].strip('/')
output_dir = sys.argv[3].strip('/')

dset = root_dir.strip('/').split('/')[-1]

for speaker_dir in sorted(glob.glob(f'{root_dir}/*')):
	speaker_id = speaker_dir.strip('/').split('/')[-1]

	for wav_path in sorted(glob.glob(f'{speaker_dir}/*.wav')):
		wav_filename = wav_path.split('/')[-1][:-4]
		tg_path = f'{tg_dir}/{speaker_id}/{wav_filename}.TextGrid'
		
		audio = AudioSegment.from_file(wav_path, format='wav', frame_rate=16000)
		tg = tgt.io.read_textgrid(tg_path)
		intervals = tg.get_tier_by_name('words').intervals
		
		for interval in intervals:
			audio_seg = audio[interval.start_time*1000:interval.end_time*1000]
			
			dest_dir = f'{output_dir}/{interval.text}'

			if not os.path.isdir(dest_dir):
				os.makedirs(dest_dir)
			
			seg_path = f'{dest_dir}/{wav_filename}_{interval.start_time}-{interval.end_time}.wav'
			print(seg_path)
			audio_seg.export(seg_path, format='wav')
				