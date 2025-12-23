import numpy, os, random, soundfile, torch, json
import re
from collections import defaultdict

def init_loader(args):
	args.trainLoader = torch.utils.data.DataLoader(train_loader(set_type = 'train', **vars(args)), batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	args.valLoader   = torch.utils.data.DataLoader(train_loader(set_type = 'val', **vars(args)),  batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = True)
	args.infLoader  = torch.utils.data.DataLoader(inf_loader(args.data_list), batch_size = 1, shuffle = False, num_workers = 0, drop_last = False)
	return args

def load_audio_all(path):
	if not os.path.isfile(path):
		print('NOT FILE', path)
		raise FileNotFoundError(f"Audio file not found: {path}")
	audio, _ = soundfile.read(path)
	return audio

def load_visual_all(path):
	if not os.path.isfile(path):
		print('NOT FILE', path)
		raise FileNotFoundError(f"Visual file not found: {path}")
	face = numpy.load(path)
	return face

class train_loader(object):
	def __init__(self, set_type, data_list, visual_path, audio_path, length, musan_path, **kwargs):
		self.length = length
		
		# Read data_list and filter central_crops only
		lines = open(data_list).read().splitlines()
		central_crops_paths = []
		for line in lines:
			line = line.replace('\n', '').strip()
			if 'central_crops' in line and line.endswith('.wav'):
				central_crops_paths.append(line)
		
		# Group by session
		session_dict = defaultdict(list)
		for path in central_crops_paths:
			# Extract session ID from path
			session_match = re.search(r'session_\d+', path)
			if session_match:
				session_id = session_match.group()
				session_dict[session_id].append(path)
		
		# Shuffle sessions randomly
		session_list = list(session_dict.keys())
		random.shuffle(session_list)
		
		# Create pairs: each session with the next session
		# TODO: Currently using first track (track_00) temporarily. Should be changed to random track selection in the future.
		self.pair_list = []
		for i in range(0, len(session_list) - 1, 2):
			session1 = session_list[i]
			session2 = session_list[i + 1]
			
			# Get first track (track_00) from each session
			# Find track_00_lip.av.wav paths
			session1_paths = [p for p in session_dict[session1] if 'track_00_lip.av.wav' in p]
			session2_paths = [p for p in session_dict[session2] if 'track_00_lip.av.wav' in p]
			
			if len(session1_paths) > 0 and len(session2_paths) > 0:
				# Randomly select one path from each session
				path1 = random.choice(session1_paths)
				path2 = random.choice(session2_paths)
				self.pair_list.append((path1, path2))
		
		print(f"Created {len(self.pair_list)} pairs from {len(session_list)} sessions")

	def __getitem__(self, index):         
		# Get pair of paths
		audio_path1, audio_path2 = self.pair_list[index]
		
		# Get metadata paths
		# Extract base path up to session directory
		parts1 = audio_path1.split('/')
		parts2 = audio_path2.split('/')
		session_idx1 = next((i for i, p in enumerate(parts1) if p.startswith('session_')), None)
		session_idx2 = next((i for i, p in enumerate(parts2) if p.startswith('session_')), None)
		
		if session_idx1 is None or session_idx2 is None:
			raise ValueError(f"Could not find session in paths: {audio_path1}, {audio_path2}")
		
		metadata_path1 = '/'.join(parts1[:session_idx1 + 1]) + '/metadata.json'
		metadata_path2 = '/'.join(parts2[:session_idx2 + 1]) + '/metadata.json'
		
		with open(metadata_path1, 'r') as file:
			metadata_dict1 = json.load(file)
		with open(metadata_path2, 'r') as file:
			metadata_dict2 = json.load(file)
		
		# Extract track number from paths (e.g., track_00 from track_00_lip.av.wav)
		track_match1 = re.search(r'track_\d+', audio_path1)
		track_match2 = re.search(r'track_\d+', audio_path2)
		track1 = track_match1.group() if track_match1 else 'track_00'
		track2 = track_match2.group() if track_match2 else 'track_00'
		
		# Get session IDs
		session_match1 = re.search(r'session_\d+', audio_path1)
		session_match2 = re.search(r'session_\d+', audio_path2)
		session1 = session_match1.group() if session_match1 else None
		session2 = session_match2.group() if session_match2 else None
		
		# Get all speakers from each session
		spk_ids1 = [spk_id for spk_id in metadata_dict1.keys() if spk_id.startswith('spk_')]
		spk_ids2 = [spk_id for spk_id in metadata_dict2.keys() if spk_id.startswith('spk_')]
		
		# Get vid_start for timing (use first speaker's timing as reference)
		if len(spk_ids1) > 0:
			vid_start1 = float(metadata_dict1[spk_ids1[0]]['central']['uem']['start'])
		else:
			vid_start1 = 0.0
		
		if len(spk_ids2) > 0:
			vid_start2 = float(metadata_dict2[spk_ids2[0]]['central']['uem']['start'])
		else:
			vid_start2 = 0.0
		
		# Use same timestamp for both mixtures (Option A)
		all_length = 28.0
		rand_num = random.random()
		start_timestamp = int(rand_num * (all_length - self.length)) + 1
		start_face1 = int(vid_start1 * 25) + int(start_timestamp * 25)
		start_face2 = int(vid_start2 * 25) + int(start_timestamp * 25)
		start_audio1 = start_face1 * 640
		start_audio2 = start_face2 * 640
		
		# Load mixture1 and mixture2
		audio1 = load_audio_all(path = audio_path1)
		audio2 = load_audio_all(path = audio_path2)
		audio1 = audio1[start_audio1:start_audio1 + int(self.length * 16000)]
		audio2 = audio2[start_audio2:start_audio2 + int(self.length * 16000)]
		
		if len(audio1) == 0:
			print('AUDIO1', audio_path1)
			raise ValueError(f"Empty audio segment: {audio_path1}")
		
		if len(audio2) == 0:
			print('AUDIO2', audio_path2)
			raise ValueError(f"Empty audio segment: {audio_path2}")
		
		# Normalize audio
		audio1_max = numpy.max(numpy.abs(audio1))
		audio2_max = numpy.max(numpy.abs(audio2))
		if audio1_max > 0:
			audio1 = numpy.divide(audio1, audio1_max)
		if audio2_max > 0:
			audio2 = numpy.divide(audio2, audio2_max)
		
		# Get lip crops for all speakers in session1 (same track as mixture)
		mixture1_lip_crops = []
		# Extract base path (up to session directory)
		base_path1 = '/'.join(audio_path1.split('/')[:audio_path1.split('/').index(session1) + 1])
		for spk_id in spk_ids1:
			# Construct lip crop path: same track as mixture
			# Format: /path/to/session_XX/speakers/spk_Y/central_crops/track_ZZ_lip.av.npy
			lip_path = os.path.join(base_path1, 'speakers', spk_id, 'central_crops', f'{track1}_lip.av.npy')
			
			if os.path.isfile(lip_path):
				face = load_visual_all(path = lip_path)
				face = face[start_face1:start_face1 + int(self.length * 25)]
				mixture1_lip_crops.append(torch.FloatTensor(face))
			else:
				print(f'LIP CROP NOT FOUND: {lip_path}')
		
		# Get lip crops for all speakers in session2 (same track as mixture)
		mixture2_lip_crops = []
		# Extract base path (up to session directory)
		base_path2 = '/'.join(audio_path2.split('/')[:audio_path2.split('/').index(session2) + 1])
		for spk_id in spk_ids2:
			# Construct lip crop path: same track as mixture
			# Format: /path/to/session_XX/speakers/spk_Y/central_crops/track_ZZ_lip.av.npy
			lip_path = os.path.join(base_path2, 'speakers', spk_id, 'central_crops', f'{track2}_lip.av.npy')
			
			if os.path.isfile(lip_path):
				face = load_visual_all(path = lip_path)
				face = face[start_face2:start_face2 + int(self.length * 25)]
				mixture2_lip_crops.append(torch.FloatTensor(face))
			else:
				print(f'LIP CROP NOT FOUND: {lip_path}')
		
		return torch.FloatTensor(audio1), \
			   mixture1_lip_crops, \
			   torch.FloatTensor(audio2), \
			   mixture2_lip_crops

	def __len__(self):
		return len(self.pair_list)

class inf_loader(object):
	def __init__(self, data_list):
		self.data_list = []
		lines = open(data_list).read().splitlines()
		for line in lines:
			self.data_list.append(line.replace('\n', ''))
		
	def __getitem__(self, index):        
		audio_path = self.data_list[index]
		visual_path = audio_path.replace('.wav', '.npy')
		audio = load_audio_all(path = audio_path)
		face = load_visual_all(path = visual_path)
		scale = numpy.max(numpy.abs(audio))
		if scale > 0:
			audio = numpy.divide(audio, scale)
		
		others = {
			'audio_path': audio_path,
			'scale': scale,
		}
		return torch.FloatTensor(audio), torch.FloatTensor(face), others

	def __len__(self):
		return len(self.data_list)

