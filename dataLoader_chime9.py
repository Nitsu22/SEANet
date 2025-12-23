import numpy, os, random, soundfile, torch, json

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
		self.data_list = []
		# data_list format: <ego_audio_path>
		lines = open(data_list).read().splitlines()
		for line in lines:
			self.data_list.append(line.replace('\n', ''))

	def __getitem__(self, index):         
		label_audio_path = self.data_list[index]
		audio_path = label_audio_path.replace('ego_crops', 'central_crops')
		visual_path = audio_path.replace('.wav', '.npy') # from central crops
		metadata_path = '/'.join(label_audio_path.split('/', 10)[:10]) + '/metadata.json'
		with open(metadata_path, 'r') as file:
			metadata_dict = json.load(file)

		spk_id = label_audio_path.split('/')[11]
		label_audio_start = float(metadata_dict[spk_id]['ego']['uem']['start']) * 16000
		vid_start = float(metadata_dict[spk_id]['central']['uem']['start'])
		all_length = 28.0
		rand_num = random.random()
		start_timestamp = int(rand_num * (all_length - self.length)) + 1
		start_face = int(vid_start * 25) + int(start_timestamp * 25)
		start_audio = start_face * 640
		start_label = int(label_audio_start) + int(start_timestamp * 16000)

		audio = load_audio_all(path = audio_path)
		face = load_visual_all(path = visual_path)
		label = load_audio_all(path = label_audio_path)
		audio = audio[start_audio:start_audio + int(self.length * 16000)]
		label = label[start_label:start_label + int(self.length * 16000)]
		
		if len(audio) == 0:
			print('AUDIO', audio_path)
			raise ValueError(f"Empty audio segment: {audio_path}")

		if len(label) == 0:
			print('LABEL AUDIO', label_audio_path)
			raise ValueError(f"Empty label audio segment: {label_audio_path}")

		noise = audio - label
		face = face[start_face:start_face + int(self.length * 25)]
		audio_max = numpy.max(numpy.abs(audio))
		label_max = numpy.max(numpy.abs(label))
		noise_max = numpy.max(numpy.abs(noise))
		if audio_max > 0:
			audio = numpy.divide(audio, audio_max)
		if label_max > 0:
			label = numpy.divide(label, label_max)
		if noise_max > 0:
			noise = numpy.divide(noise, noise_max)
		
		return torch.FloatTensor(audio), \
			   torch.FloatTensor(face), \
			   torch.FloatTensor(label), \
			   torch.FloatTensor(noise)

	def __len__(self):
		return len(self.data_list)

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

