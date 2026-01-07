import numpy, os, torch
from dataLoader_chime9_mixit_all import load_audio_all, load_visual_all

class inf_loader_session_speakers(object):
	"""1つのセッションの全話者を推論するためのローダー（mixtureは足さない）"""
	def __init__(self, data_list, session_id='session_00', track='track_00'):
		"""
		Args:
			data_list: データリストファイルのパス
			session_id: 推論するセッションID（デフォルト: 'session_00'）
			track: 使用するトラック（デフォルト: 'track_00'）
		"""
		# Read data_list and filter central_crops only
		lines = open(data_list).read().splitlines()
		central_crops_paths = []
		for line in lines:
			line = line.replace('\n', '').strip()
			if 'central_crops' in line and line.endswith('.wav'):
				central_crops_paths.append(line)
		
		# Find session paths
		session_paths = [p for p in central_crops_paths if session_id in p and f'{track}_lip.av.wav' in p]
		
		if len(session_paths) == 0:
			raise ValueError(f"No paths found for {session_id} with {track}_lip.av.wav")
		
		# Get session base path from first path
		parts = session_paths[0].split('/')
		session_idx = next((i for i, p in enumerate(parts) if p.startswith('session_')), None)
		
		if session_idx is None:
			raise ValueError(f"Could not find session in path: {session_paths[0]}")
		
		session = parts[session_idx]
		base_path = '/'.join(parts[:session_idx + 1])
		metadata_path = os.path.join(base_path, 'metadata.json')
		
		# Load metadata to get all speakers
		import json
		with open(metadata_path, 'r') as file:
			metadata_dict = json.load(file)
		
		spk_ids = [spk_id for spk_id in metadata_dict.keys() if spk_id.startswith('spk_')]
		spk_ids.sort()  # Sort for consistent ordering
		
		# Use first speaker's audio file as mixture (single session mixture, not summed)
		# This represents the mixture audio for the session (not summed with another session)
		mixture_audio_path = session_paths[0]  # Use first speaker's file as mixture
		
		# Load mixture audio
		mixture_audio = load_audio_all(path=mixture_audio_path)
		
		# Normalize and save scale (same as inf_loader)
		scale = numpy.max(numpy.abs(mixture_audio))
		if scale > 0:
			mixture_audio_normalized = numpy.divide(mixture_audio, scale)
		else:
			mixture_audio_normalized = mixture_audio
			scale = 1.0
		
		self.mixture_audio = mixture_audio_normalized
		self.mixture_audio_original = mixture_audio  # Keep original for output
		self.mixture_audio_path = mixture_audio_path
		self.base_path = base_path
		self.session_id = session
		self.track = track
		self.spk_ids = spk_ids
		self.scale = scale
		
		print(f"Found {len(spk_ids)} speakers in {session_id}: {spk_ids}")
		print(f"Using mixture audio (single session, not summed): {mixture_audio_path}")
		print(f"Mixture audio length: {len(mixture_audio)} samples ({len(mixture_audio)/16000:.2f} seconds)")
	
	def __getitem__(self, index):
		"""各話者ごとにデータを返す"""
		if index >= len(self.spk_ids):
			raise IndexError(f"Index {index} out of range for {len(self.spk_ids)} speakers")
		
		spk_id = self.spk_ids[index]
		
		# Use mixture audio (already normalized)
		audio = self.mixture_audio.copy()
		
		# Load speaker's lip crop
		lip_path = os.path.join(self.base_path, 'speakers', spk_id, 'central_crops', f'{self.track}.npy')
		if not os.path.isfile(lip_path):
			raise FileNotFoundError(f"Lip file not found: {lip_path}")
		
		face = load_visual_all(path=lip_path)
		
		# Ensure audio and face have compatible lengths
		# Audio: 16000 Hz, Face: 25 fps
		# 1 frame of face = 640 samples of audio
		face_len = face.shape[0]
		audio_len = len(audio)
		expected_audio_len = face_len * 640
		
		# Adjust audio length to match face length
		if audio_len > expected_audio_len:
			audio = audio[:expected_audio_len]
		elif audio_len < expected_audio_len:
			# Pad with zeros
			audio = numpy.concatenate([audio, numpy.zeros(expected_audio_len - audio_len, dtype=audio.dtype)])
		
		# Ensure face has correct shape [F, C]
		if face.ndim == 1:
			face = face.reshape(-1, 1)
		elif face.ndim > 2:
			raise ValueError(f"Unexpected face shape: {face.shape}")
		
		others = {
			'audio_path': self.mixture_audio_path,  # DataLoader will convert to list
			'scale': self.scale,  # DataLoader will convert to list
			'spk_id': spk_id,  # DataLoader will convert to list
			'session_id': self.session_id,  # DataLoader will convert to list
		}
		
		return torch.FloatTensor(audio), torch.FloatTensor(face), others
	
	def __len__(self):
		return len(self.spk_ids)

class inf_loader_session_pair(object):
	"""2つのセッションを混ぜて推論するためのローダー（学習と同様に）"""
	def __init__(self, data_list, session_id1='session_00', session_id2='session_01', track='track_00'):
		"""
		Args:
			data_list: データリストファイルのパス
			session_id1: 1つ目のセッションID
			session_id2: 2つ目のセッションID
			track: 使用するトラック
		"""
		lines = open(data_list).read().splitlines()
		central_crops_paths = []
		for line in lines:
			line = line.replace('\n', '').strip()
			if 'central_crops' in line and line.endswith('.wav'):
				central_crops_paths.append(line)
		
		# Find session paths
		session1_paths = [p for p in central_crops_paths if session_id1 in p and f'{track}_lip.av.wav' in p]
		session2_paths = [p for p in central_crops_paths if session_id2 in p and f'{track}_lip.av.wav' in p]
		
		if len(session1_paths) == 0:
			raise ValueError(f"No paths found for {session_id1} with {track}_lip.av.wav")
		if len(session2_paths) == 0:
			raise ValueError(f"No paths found for {session_id2} with {track}_lip.av.wav")
		
		# Use first path from each session as mixture audio
		audio_path1 = session1_paths[0]
		audio_path2 = session2_paths[0]
		
		# Extract session IDs and metadata paths
		parts1 = audio_path1.split('/')
		parts2 = audio_path2.split('/')
		session_idx1 = next((i for i, p in enumerate(parts1) if p.startswith('session_')), None)
		session_idx2 = next((i for i, p in enumerate(parts2) if p.startswith('session_')), None)
		
		if session_idx1 is None or session_idx2 is None:
			raise ValueError(f"Could not find session in paths")
		
		session1 = parts1[session_idx1]
		session2 = parts2[session_idx2]
		metadata_path1 = '/'.join(parts1[:session_idx1 + 1]) + '/metadata.json'
		metadata_path2 = '/'.join(parts2[:session_idx2 + 1]) + '/metadata.json'
		
		import json
		with open(metadata_path1, 'r') as file:
			metadata_dict1 = json.load(file)
		with open(metadata_path2, 'r') as file:
			metadata_dict2 = json.load(file)
		
		# Get all speakers from each session
		spk_ids1 = [spk_id for spk_id in metadata_dict1.keys() if spk_id.startswith('spk_')]
		spk_ids2 = [spk_id for spk_id in metadata_dict2.keys() if spk_id.startswith('spk_')]
		spk_ids1.sort()
		spk_ids2.sort()
		
		# Load mixture audios (full length, not segmented)
		audio1 = load_audio_all(path=audio_path1)
		audio2 = load_audio_all(path=audio_path2)
		
		# Normalize (same as training)
		audio1_max = numpy.max(numpy.abs(audio1))
		audio2_max = numpy.max(numpy.abs(audio2))
		if audio1_max > 0:
			audio1_normalized = numpy.divide(audio1, audio1_max)
		else:
			audio1_normalized = audio1
			audio1_max = 1.0
		if audio2_max > 0:
			audio2_normalized = numpy.divide(audio2, audio2_max)
		else:
			audio2_normalized = audio2
			audio2_max = 1.0
		
		# Create mixture (sum of both sessions, same as training: mixture_plus = audio1 + audio2)
		min_length = min(len(audio1_normalized), len(audio2_normalized))
		mixture_audio = audio1_normalized[:min_length] + audio2_normalized[:min_length]
		
		# Store original audios for output (denormalized)
		self.mixture_audio = mixture_audio
		self.mixture_audio_original = mixture_audio * max(audio1_max, audio2_max)  # For output
		self.mixture_audio1_original = audio1  # Original audio1
		self.mixture_audio2_original = audio2  # Original audio2
		self.base_path1 = '/'.join(parts1[:session_idx1 + 1])
		self.base_path2 = '/'.join(parts2[:session_idx2 + 1])
		self.session_id1 = session1
		self.session_id2 = session2
		self.track = track
		self.spk_ids1 = spk_ids1
		self.spk_ids2 = spk_ids2
		self.scale1 = audio1_max
		self.scale2 = audio2_max
		self.mixture_scale = max(audio1_max, audio2_max)
		self.min_length = min_length
		
		# Create list of all speakers with session info
		self.speaker_list = []
		for spk_id in spk_ids1:
			self.speaker_list.append((session1, spk_id, 1))  # session, spk_id, session_index
		for spk_id in spk_ids2:
			self.speaker_list.append((session2, spk_id, 2))
		
		print(f"Session pair: {session_id1} + {session_id2}")
		print(f"Session1 ({session_id1}): {len(spk_ids1)} speakers - {spk_ids1}")
		print(f"Session2 ({session_id2}): {len(spk_ids2)} speakers - {spk_ids2}")
		print(f"Mixture audio length: {min_length} samples ({min_length/16000:.2f} seconds)")
	
	def __getitem__(self, index):
		"""各話者ごとにデータを返す"""
		if index >= len(self.speaker_list):
			raise IndexError(f"Index {index} out of range for {len(self.speaker_list)} speakers")
		
		session_id, spk_id, session_idx = self.speaker_list[index]
		
		# Use mixture audio (already normalized and summed)
		audio = self.mixture_audio.copy()
		
		# Determine which session this speaker belongs to
		if session_idx == 1:
			base_path = self.base_path1
		else:
			base_path = self.base_path2
		
		# Load speaker's lip crop
		lip_path = os.path.join(base_path, 'speakers', spk_id, 'central_crops', f'{self.track}.npy')
		if not os.path.isfile(lip_path):
			raise FileNotFoundError(f"Lip file not found: {lip_path}")
		
		face = load_visual_all(path=lip_path)
		
		# Ensure audio and face have compatible lengths
		face_len = face.shape[0]
		audio_len = len(audio)
		expected_audio_len = face_len * 640
		
		# Adjust audio length to match face length
		if audio_len > expected_audio_len:
			audio = audio[:expected_audio_len]
		elif audio_len < expected_audio_len:
			audio = numpy.concatenate([audio, numpy.zeros(expected_audio_len - audio_len, dtype=audio.dtype)])
		
		# Ensure face has correct shape [F, C]
		if face.ndim == 1:
			face = face.reshape(-1, 1)
		elif face.ndim > 2:
			raise ValueError(f"Unexpected face shape: {face.shape}")
		
		# Use mixture_scale for denormalization
		others = {
			'session_id': session_id,
			'spk_id': spk_id,
			'scale': self.mixture_scale,
		}
		
		return torch.FloatTensor(audio), torch.FloatTensor(face), others
	
	def __len__(self):
		return len(self.speaker_list)

