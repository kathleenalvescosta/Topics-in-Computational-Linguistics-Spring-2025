import os,re

filedir = '/home/kathleencosta/cv-corpus-10.0-delta-2022-07-04-pt/pt/'

filenames = os.listdir(filedir+'clips')

os.system('mkdir ' + filedir + 'mhwav')

for filename in filenames:
	if re.search('\.mp3$',filename):
		newname = filename[:-3] + 'wav'
		os.system(
			'ffmpeg -i ' + filedir + 'clips/' + \
			filename + ' ' + filedir + \
			'mhwav/' + newname
		)

