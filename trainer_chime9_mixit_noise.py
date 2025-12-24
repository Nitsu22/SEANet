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
		# MixIT: batchサイズ1前提で各話者出力を合計
		B, time_start, nloss = 1, time.time(), 0
		nloss_s_main, nloss_n_main = 0, 0
		nloss_s_rest, nloss_n_rest = 0, 0
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		lr = self.optim.param_groups[0]['lr']	
		for num, (audio1, mixture1_lip_crops, audio2, mixture2_lip_crops) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			with autocast():				
				# Ensure batch dimension
				audio1 = audio1.cuda()
				audio2 = audio2.cuda()
				if audio1.dim() == 1:
					audio1 = audio1.unsqueeze(0)
				if audio2.dim() == 1:
					audio2 = audio2.unsqueeze(0)

				mixture_plus = audio1 + audio2  # [1, T]

				# 話者数を記録（スキップ前）
				num_speakers_mixture1 = len(mixture1_lip_crops)
				num_speakers_mixture2 = len(mixture2_lip_crops)
				total_speakers = num_speakers_mixture1 + num_speakers_mixture2

				# 話者数が0の場合はエラー
				if num_speakers_mixture1 == 0 or num_speakers_mixture2 == 0:
					raise ValueError("Lip crops are empty for mixture1 or mixture2.")

				# 全話者の出力を保存 (out_s, out_n) のタプル、全反復含む
				all_speaker_outputs = []
				mixture1_indices = []  # mixture1の話者のインデックス
				mixture2_indices = []  # mixture2の話者のインデックス

				# mixture1の各話者に対してモデルを実行
				for lip in mixture1_lip_crops:
					lip = lip.cuda()
					if lip.dim() == 2:
						lip = lip.unsqueeze(0)  # [1, F, C]
					
					# サイズチェック（フレーム数が0の場合はスキップ）
					if lip.shape[1] == 0:
						print(f'WARNING: Empty lip crop detected in mixture1, shape={lip.shape}, skipping...')
						continue
					
					out_speech, out_n = self.model(mixture_plus, lip, M = B)
					all_speaker_outputs.append((out_speech, out_n))  # 全反復含む [6*B, T]
					mixture1_indices.append(len(all_speaker_outputs) - 1)

				# mixture2の各話者に対してモデルを実行
				for lip in mixture2_lip_crops:
					lip = lip.cuda()
					if lip.dim() == 2:
						lip = lip.unsqueeze(0)
					
					# サイズチェック（フレーム数が0の場合はスキップ）
					if lip.shape[1] == 0:
						print(f'WARNING: Empty lip crop detected in mixture2, shape={lip.shape}, skipping...')
						continue
					
					out_speech, out_n = self.model(mixture_plus, lip, M = B)
					all_speaker_outputs.append((out_speech, out_n))  # 全反復含む [6*B, T]
					mixture2_indices.append(len(all_speaker_outputs) - 1)

				# 実際に処理された話者数を確認
				if len(all_speaker_outputs) == 0:
					raise ValueError("No valid speaker outputs after processing lip crops.")

				# loss_s_main: mixture1とmixture2ごとに計算
				# mixture1内の全話者のout_s[-B:,:]を合計
				estim1_main = None
				for idx in mixture1_indices:
					out_s, _ = all_speaker_outputs[idx]
					if estim1_main is None:
						estim1_main = out_s[-B:,:]  # [1, T]
					else:
						estim1_main = estim1_main + out_s[-B:,:]
				
				# mixture2内の全話者のout_s[-B:,:]を合計
				estim2_main = None
				for idx in mixture2_indices:
					out_s, _ = all_speaker_outputs[idx]
					if estim2_main is None:
						estim2_main = out_s[-B:,:]  # [1, T]
					else:
						estim2_main = estim2_main + out_s[-B:,:]

				loss_s1_main = self.loss_se.forward(estim1_main, audio1)
				loss_s2_main = self.loss_se.forward(estim2_main, audio2)
				loss_s_main_avg = (loss_s1_main + loss_s2_main) / 2.0

				# loss_n_main: 各話者ごとに計算（その話者以外の全話者のout_s[-B:,:]の合計と比較）
				all_loss_n_main = []
				for spk_idx, (out_s, out_n) in enumerate(all_speaker_outputs):
					# その話者以外の全話者のout_s[-B:,:]を合計
					other_speakers_sum = None
					for other_idx, (other_out_s, _) in enumerate(all_speaker_outputs):
						if other_idx != spk_idx:
							if other_speakers_sum is None:
								other_speakers_sum = other_out_s[-B:,:]  # [1, T]
							else:
								other_speakers_sum = other_speakers_sum + other_out_s[-B:,:]
					
					loss_n_main = self.loss_se.forward(out_n[-B:,:], other_speakers_sum)
					# ========== DEBUG: nanチェック（後で削除する） ==========
					if torch.isnan(loss_n_main) or torch.isinf(loss_n_main):
						print(f"\nWARNING: loss_n_main is nan/inf at spk_idx={spk_idx}, loss_n_main={loss_n_main}")
						print(f"  out_n shape: {out_n[-B:,:].shape}, other_speakers_sum shape: {other_speakers_sum.shape}")
						print(f"  out_n stats: min={out_n[-B:,:].min()}, max={out_n[-B:,:].max()}, mean={out_n[-B:,:].mean()}")
						print(f"  other_speakers_sum stats: min={other_speakers_sum.min()}, max={other_speakers_sum.max()}, mean={other_speakers_sum.mean()}")
						# nanの場合はスキップして続行
						continue
					# ========== ここまで削除する ==========
					all_loss_n_main.append(loss_n_main)
				
				# ========== DEBUG: nanチェック（後で削除する） ==========
				if len(all_loss_n_main) == 0:
					print("\nWARNING: all_loss_n_main is empty, using zero tensor")
					loss_n_main_avg = torch.tensor(0.0).cuda()
				else:
					loss_n_main_avg = sum(all_loss_n_main) / len(all_loss_n_main)
				# ========== ここまで削除する ==========

				# loss_s_rest: mixture1とmixture2ごとに補助反復で計算
				# mixture1内の全話者のout_s[:-B,:]を合計
				estim1_rest = None
				for idx in mixture1_indices:
					out_s, _ = all_speaker_outputs[idx]
					if estim1_rest is None:
						estim1_rest = out_s[:-B,:]  # [5, T] (B=1の場合)
					else:
						estim1_rest = estim1_rest + out_s[:-B,:]
				
				# mixture2内の全話者のout_s[:-B,:]を合計
				estim2_rest = None
				for idx in mixture2_indices:
					out_s, _ = all_speaker_outputs[idx]
					if estim2_rest is None:
						estim2_rest = out_s[:-B,:]  # [5, T] (B=1の場合)
					else:
						estim2_rest = estim2_rest + out_s[:-B,:]

				loss_s1_rest = self.loss_se.forward(estim1_rest, audio1.repeat(5, 1))
				loss_s2_rest = self.loss_se.forward(estim2_rest, audio2.repeat(5, 1))
				loss_s_rest_avg = (loss_s1_rest + loss_s2_rest) / 2.0

				# loss_n_rest: 各話者ごとに補助反復で計算（その話者以外の全話者のout_s[:-B,:]の合計と比較）
				all_loss_n_rest = []
				for spk_idx, (out_s, out_n) in enumerate(all_speaker_outputs):
					# その話者以外の全話者のout_s[:-B,:]を合計
					other_speakers_rest_sum = None
					for other_idx, (other_out_s, _) in enumerate(all_speaker_outputs):
						if other_idx != spk_idx:
							if other_speakers_rest_sum is None:
								other_speakers_rest_sum = other_out_s[:-B,:]  # [5, T] (B=1の場合)
							else:
								other_speakers_rest_sum = other_speakers_rest_sum + other_out_s[:-B,:]
					
					loss_n_rest = self.loss_se.forward(out_n[:-B,:], other_speakers_rest_sum)
					# ========== DEBUG: nanチェック（後で削除する） ==========
					if torch.isnan(loss_n_rest) or torch.isinf(loss_n_rest):
						print(f"\nWARNING: loss_n_rest is nan/inf at spk_idx={spk_idx}, loss_n_rest={loss_n_rest}")
						# nanの場合はスキップして続行
						continue
					# ========== ここまで削除する ==========
					all_loss_n_rest.append(loss_n_rest)
				
				# ========== DEBUG: nanチェック（後で削除する） ==========
				if len(all_loss_n_rest) == 0:
					print("\nWARNING: all_loss_n_rest is empty, using zero tensor")
					loss_n_rest_avg = torch.tensor(0.0).cuda()
				else:
					loss_n_rest_avg = sum(all_loss_n_rest) / len(all_loss_n_rest)
				# ========== ここまで削除する ==========

				# 最終損失
				loss = loss_s_main_avg + (loss_n_main_avg + loss_n_rest_avg + loss_s_rest_avg) * args.alpha

			scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
			scaler.step(self.optim)
			scaler.update()

			nloss += loss.detach().cpu().numpy()
			nloss_s_main += loss_s_main_avg.detach().cpu().numpy()
			nloss_n_main += loss_n_main_avg.detach().cpu().numpy()
			nloss_s_rest += loss_s_rest_avg.detach().cpu().numpy()
			nloss_n_rest += loss_n_rest_avg.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.3f (s_main: %.3f, n_main: %.3f, s_rest: %.3f, n_rest: %.3f)\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, nloss/num, nloss_s_main/num, nloss_n_main/num, nloss_s_rest/num, nloss_n_rest/num))
			sys.stderr.flush()


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