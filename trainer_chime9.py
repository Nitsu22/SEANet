import torch, sys, os, time
import torch.nn as nn
from tools import *
from loss  import *
from model.dprnn import dprnn
from model.muse import muse
from model.avsep import avsep
from model.seanet import seanet
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict
import soundfile
import numpy 

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.init_model != "":
		print("Model %s loaded from pretrain!"%args.init_model)
		s.load_parameters(args.init_model)
	elif len(args.modelfiles) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles[-1])
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		if args.backbone == 'seanet':
			self.model       = seanet(256, 40, 64, 128, 100, 6).cuda()
		elif args.backbone == 'avsep':
			self.model       = avsep().cuda()
		elif args.backbone == 'muse':
			self.model       = muse(M = 800).cuda() # Based on your dataset
		elif args.backbone == 'dprnn':
			self.model       = dprnn().cuda()
		self.loss_se     = loss_speech().cuda()
		self.optim       = torch.optim.AdamW(self.parameters(), lr = args.lr)
		self.scheduler   = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.val_step, gamma = args.lr_decay)
		print("Model para number = %.2f"%(sum(param.numel() for param in self.parameters()) / 1e6))
		
	def train_network(self, args):
		B, time_start, nloss = args.batch_size, time.time(), 0
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		lr = self.optim.param_groups[0]['lr']	
		for num, (audio, face, speech, noise) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			with autocast():				
				audio, face, speech, noise = audio.cuda(), face.cuda(), speech.cuda(), noise.cuda()
				if args.backbone == 'seanet':
					out_s, out_n = self.model(audio, face, M = B)			
					loss_s_main = self.loss_se.forward(out_s[-B:,:], speech)
					loss_n_main = self.loss_se.forward(out_n[-B:,:], noise)	
					loss_n_rest = self.loss_se.forward(out_n[:-B,:], noise.repeat(5, 1))
					loss_s_rest = self.loss_se.forward(out_s[:-B,:], speech.repeat(5, 1))
					loss = loss_s_main + (loss_n_main + loss_n_rest + loss_s_rest) * args.alpha

			scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
			scaler.step(self.optim)
			scaler.update()

			nloss += loss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/B/num))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/B/num))
		args.score_file.flush()
		return

	def eval_network(self, eval_type, args):
		Loader = args.valLoader   if eval_type == 'Val' else args.infLoader
		B      = args.batch_size  if eval_type == 'Val' else 1
		self.eval()
		time_start = time.time()
		for num, (audio, face, others) in enumerate(Loader, start = 1):
			self.zero_grad()
			with torch.no_grad():
				audio, face = audio.cuda(), face.cuda()
				audio_seg = 8
				audio_len = audio.shape[1]
				face_len = face.shape[1]
				output_segments = []
				for seg in range(audio_seg):
					start_audio = seg * (audio_len // audio_seg)
					start_face = seg * (face_len // audio_seg)
					if seg < (audio_seg - 1):
						end_audio = start_audio + (audio_len // audio_seg)
						end_face = start_face + (face_len // audio_seg)
					else:
						end_audio = audio_len
						end_face = face_len
					if args.backbone == 'seanet':
						out_speech, _ = self.model(audio[:, start_audio:end_audio], 
						                           face[:, start_face:end_face, :], B)
						out = out_speech[-B:,:]
					
					output_segments.append(out)
				out_cat = torch.cat(output_segments, dim=1)

			time_used = time.time() - time_start

			audio_path = others['audio_path'][0]
			scale = others['scale'][0]
			out_wav_path = audio_path.replace('.wav', '_clean_ft.wav')
			soundfile.write(out_wav_path, numpy.multiply(out_cat[0].cpu(), scale), 16000)
		return

	def save_parameters(self, path):
		model = OrderedDict(list(self.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		selfState = self.state_dict()
		loadedState = torch.load(path)	
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				name = 'model.' + name
				if name not in selfState:
					print("%s is not in the model."%origName)
					continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
				continue
			selfState[name].copy_(param)