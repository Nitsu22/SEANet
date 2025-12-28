import argparse, os
import numpy
import soundfile
from tools import *
from trainer_chime9_mixit import *
from dataLoader_chime9_mixit_inference import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser(description = "Audio-visual target speaker extraction inference for session pair (MixIT).")

parser.add_argument('--batch_size', type=int,   default=1,        help='Batch size for inference')
parser.add_argument('--n_cpu',      type=int,   default=0,        help='Number of loader threads')
parser.add_argument('--init_model', type=str,   default="",       help='Path to model file')
parser.add_argument('--data_list',  type=str,   default="",       help='The path of the data list')
parser.add_argument('--backbone',   type=str,   default="seanet", help='Model backbone (seanet)')
parser.add_argument('--session_id1', type=str,   default="session_00", help='First session ID')
parser.add_argument('--session_id2', type=str,   default="session_01", help='Second session ID')
parser.add_argument('--track',      type=str,   default="track_00", help='Track to use')
parser.add_argument('--output_dir', type=str,   default="",       help='Output directory for separated audio files')
parser.add_argument('--save_path',  type=str,   default="",       help='Path for save_path (used by init_system)')
parser.add_argument('--lr',         type=float, default=0.0010,   help='Init learning rate (not used in inference)')
parser.add_argument('--val_step',   type=int,   default=3,        help='Validation step (not used in inference)')
parser.add_argument("--lr_decay",   type=float, default=0.97,     help='Learning rate decay (not used in inference)')

args = init_system(parser.parse_args())

# Load model
s = init_trainer(args)
if args.init_model == "":
	raise ValueError("--init_model must be specified")
print(f"Loading model from: {args.init_model}")
s.load_parameters(args.init_model)

# Create inference loader
inf_loader_obj = inf_loader_session_pair(args.data_list, session_id1=args.session_id1, session_id2=args.session_id2, track=args.track)
args.infLoader = torch.utils.data.DataLoader(
	inf_loader_obj,
	batch_size=args.batch_size,
	shuffle=False,
	num_workers=args.n_cpu,
	drop_last=False
)

# Create output directory
if args.output_dir == "":
	args.output_dir = os.path.join(args.save_path, 'inference', f"{args.session_id1}_{args.session_id2}")
os.makedirs(args.output_dir, exist_ok=True)

# Save mixture audios
mixture_output_path = os.path.join(args.output_dir, f"mixture_{args.track}_input.wav")
soundfile.write(mixture_output_path, inf_loader_obj.mixture_audio_original, 16000)
print(f"Saved mixture (input) audio to: {mixture_output_path}")

# Save original session audios
mixture1_output_path = os.path.join(args.output_dir, f"{args.session_id1}_{args.track}_original.wav")
mixture2_output_path = os.path.join(args.output_dir, f"{args.session_id2}_{args.track}_original.wav")
soundfile.write(mixture1_output_path, inf_loader_obj.mixture_audio1_original, 16000)
soundfile.write(mixture2_output_path, inf_loader_obj.mixture_audio2_original, 16000)

# Run inference
print(f"Starting inference for {args.session_id1} + {args.session_id2}...")
print(f"Output directory: {args.output_dir}")
print(f"Number of speakers: {len(args.infLoader)}")

s.eval()
B = args.batch_size
import time
time_start = time.time()

# Store separated audios by session
separated_audios_session1 = []
separated_audios_session2 = []
separated_audio_dict = {}

for num, (audio, face, others) in enumerate(args.infLoader, start=1):
	with torch.no_grad():
		audio, face = audio.cuda(), face.cuda()
		
		if audio.dim() == 1:
			audio = audio.unsqueeze(0)
		if face.dim() == 2:
			face = face.unsqueeze(0)
		
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
				out_speech, _ = s.model(audio[:, start_audio:end_audio], 
				                       face[:, start_face:end_face, :], B)
				out = out_speech[-B:,:]
			
			output_segments.append(out)
		
		out_cat = torch.cat(output_segments, dim=1)
	
	session_id = others['session_id'][0] if isinstance(others['session_id'], list) else others['session_id']
	spk_id = others['spk_id'][0] if isinstance(others['spk_id'], list) else others['spk_id']
	scale = others['scale'][0] if isinstance(others['scale'], list) else others['scale']
	
	output_audio = numpy.multiply(out_cat[0].cpu().numpy(), scale)
	
	# Save separated audio with session ID in filename
	output_filename = f"{session_id}_{spk_id}_{args.track}_separated.wav"
	output_path = os.path.join(args.output_dir, output_filename)
	soundfile.write(output_path, output_audio, 16000)
	
	# Store by session
	if session_id == args.session_id1:
		separated_audios_session1.append(output_audio)
	else:
		separated_audios_session2.append(output_audio)
	
	separated_audio_dict[(session_id, spk_id)] = output_audio
	
	time_used = time.time() - time_start
	print(f"Processed {num}/{len(args.infLoader)}: {session_id}_{spk_id} -> {output_path} (time: {time_used:.2f}s)")

# Sum separated audios for each session
print("\nSumming separated audios for each session...")

results_path = os.path.join(args.output_dir, "results.txt")
with open(results_path, 'w') as f:
	f.write("=" * 60 + "\n")
	f.write("SI-SNR Evaluation Results (Session Pair)\n")
	f.write("=" * 60 + "\n")
	f.write(f"Session1: {args.session_id1}\n")
	f.write(f"Session2: {args.session_id2}\n")
	f.write(f"Track: {args.track}\n")
	f.write("\n")

# Process session1
if len(separated_audios_session1) > 0:
	# Debug: Check audio lengths
	separated_lengths1 = [len(audio) for audio in separated_audios_session1]
	min_separated_length1 = min(separated_lengths1)
	mixture1_length = len(inf_loader_obj.mixture_audio1_original)
	print(f"\nSession1 audio length check:")
	print(f"  Separated audio lengths: {separated_lengths1}")
	print(f"  Min separated length: {min_separated_length1}")
	print(f"  Mixture1 original length: {mixture1_length}")
	
	# Use original mixture length as reference, pad separated audios if needed
	target_length1 = mixture1_length
	
	truncated_audios1 = []
	for audio in separated_audios_session1:
		if hasattr(audio, 'numpy'):
			audio = audio.numpy()
		elif hasattr(audio, 'cpu'):
			audio = audio.cpu().numpy()
		audio_array = numpy.asarray(audio, dtype=numpy.float32)
		# Truncate or pad to target length
		if len(audio_array) > target_length1:
			truncated_audios1.append(audio_array[:target_length1])
		elif len(audio_array) < target_length1:
			# Pad with zeros
			padded = numpy.zeros(target_length1, dtype=numpy.float32)
			padded[:len(audio_array)] = audio_array
			truncated_audios1.append(padded)
		else:
			truncated_audios1.append(audio_array)
	
	summed_audio1 = numpy.sum(numpy.array(truncated_audios1), axis=0)
	mixture1_truncated = numpy.asarray(inf_loader_obj.mixture_audio1_original, dtype=numpy.float32)
	print(f"  Final target_length1: {target_length1}")
	print(f"  Summed audio1 length: {len(summed_audio1)}")
	print(f"  Mixture1 truncated length: {len(mixture1_truncated)}")
	
	summed_output_path1 = os.path.join(args.output_dir, f"summed_{args.session_id1}_{args.track}_separated.wav")
	soundfile.write(summed_output_path1, summed_audio1, 16000)
	print(f"Saved summed separated audio for {args.session_id1} to: {summed_output_path1}")
	
	# Normalize both for SI-SNR calculation (same as training: normalized comparison)
	mixture1_max = numpy.max(numpy.abs(mixture1_truncated))
	summed_audio1_max = numpy.max(numpy.abs(summed_audio1))
	if mixture1_max > 0:
		mixture1_normalized = mixture1_truncated / mixture1_max
	else:
		mixture1_normalized = mixture1_truncated
	if summed_audio1_max > 0:
		summed_audio1_normalized = summed_audio1 / summed_audio1_max
	else:
		summed_audio1_normalized = summed_audio1
	
	mixture_tensor1 = torch.FloatTensor(mixture1_normalized).unsqueeze(0)
	summed_tensor1 = torch.FloatTensor(summed_audio1_normalized).unsqueeze(0)
	sisnr_summed1 = cal_SISNR(mixture_tensor1, summed_tensor1)
	sisnr_summed_value1 = sisnr_summed1.item()
	
	with open(results_path, 'a') as f:
		f.write("-" * 60 + "\n")
		f.write(f"Session1 ({args.session_id1}) Metrics:\n")
		f.write("-" * 60 + "\n")
		f.write(f"Number of speakers: {len(separated_audios_session1)}\n")
		f.write(f"SI-SNR (Summed Separated vs Original): {sisnr_summed_value1:.4f} dB\n")
		f.write("\n")

# Process session2
if len(separated_audios_session2) > 0:
	# Debug: Check audio lengths
	separated_lengths2 = [len(audio) for audio in separated_audios_session2]
	min_separated_length2 = min(separated_lengths2)
	mixture2_length = len(inf_loader_obj.mixture_audio2_original)
	print(f"\nSession2 audio length check:")
	print(f"  Separated audio lengths: {separated_lengths2}")
	print(f"  Min separated length: {min_separated_length2}")
	print(f"  Mixture2 original length: {mixture2_length}")
	
	# Use original mixture length as reference, pad separated audios if needed
	target_length2 = mixture2_length
	
	truncated_audios2 = []
	for audio in separated_audios_session2:
		if hasattr(audio, 'numpy'):
			audio = audio.numpy()
		elif hasattr(audio, 'cpu'):
			audio = audio.cpu().numpy()
		audio_array = numpy.asarray(audio, dtype=numpy.float32)
		# Truncate or pad to target length
		if len(audio_array) > target_length2:
			truncated_audios2.append(audio_array[:target_length2])
		elif len(audio_array) < target_length2:
			# Pad with zeros
			padded = numpy.zeros(target_length2, dtype=numpy.float32)
			padded[:len(audio_array)] = audio_array
			truncated_audios2.append(padded)
		else:
			truncated_audios2.append(audio_array)
	
	summed_audio2 = numpy.sum(numpy.array(truncated_audios2), axis=0)
	mixture2_truncated = numpy.asarray(inf_loader_obj.mixture_audio2_original, dtype=numpy.float32)
	print(f"  Final target_length2: {target_length2}")
	print(f"  Summed audio2 length: {len(summed_audio2)}")
	print(f"  Mixture2 truncated length: {len(mixture2_truncated)}")
	
	summed_output_path2 = os.path.join(args.output_dir, f"summed_{args.session_id2}_{args.track}_separated.wav")
	soundfile.write(summed_output_path2, summed_audio2, 16000)
	print(f"Saved summed separated audio for {args.session_id2} to: {summed_output_path2}")
	
	# Normalize both for SI-SNR calculation (same as training: normalized comparison)
	mixture2_max = numpy.max(numpy.abs(mixture2_truncated))
	summed_audio2_max = numpy.max(numpy.abs(summed_audio2))
	if mixture2_max > 0:
		mixture2_normalized = mixture2_truncated / mixture2_max
	else:
		mixture2_normalized = mixture2_truncated
	if summed_audio2_max > 0:
		summed_audio2_normalized = summed_audio2 / summed_audio2_max
	else:
		summed_audio2_normalized = summed_audio2
	
	mixture_tensor2 = torch.FloatTensor(mixture2_normalized).unsqueeze(0)
	summed_tensor2 = torch.FloatTensor(summed_audio2_normalized).unsqueeze(0)
	sisnr_summed2 = cal_SISNR(mixture_tensor2, summed_tensor2)
	sisnr_summed_value2 = sisnr_summed2.item()
	
	with open(results_path, 'a') as f:
		f.write("-" * 60 + "\n")
		f.write(f"Session2 ({args.session_id2}) Metrics:\n")
		f.write("-" * 60 + "\n")
		f.write(f"Number of speakers: {len(separated_audios_session2)}\n")
		f.write(f"SI-SNR (Summed Separated vs Original): {sisnr_summed_value2:.4f} dB\n")
		f.write("\n")

# Write output files list
with open(results_path, 'a') as f:
	f.write("-" * 60 + "\n")
	f.write("Output Files:\n")
	f.write("-" * 60 + "\n")
	f.write(f"Mixture (input): {os.path.basename(mixture_output_path)}\n")
	f.write(f"{args.session_id1} original: {os.path.basename(mixture1_output_path)}\n")
	f.write(f"{args.session_id2} original: {os.path.basename(mixture2_output_path)}\n")
	if len(separated_audios_session1) > 0:
		f.write(f"Summed {args.session_id1}: {os.path.basename(summed_output_path1)}\n")
	if len(separated_audios_session2) > 0:
		f.write(f"Summed {args.session_id2}: {os.path.basename(summed_output_path2)}\n")
	f.write("\nSeparated files:\n")
	for (session_id, spk_id) in sorted(separated_audio_dict.keys()):
		f.write(f"{session_id}_{spk_id}_{args.track}_separated.wav\n")

print(f"\nResults saved to: {results_path}")
print(f"\nInference completed! Output files saved to: {args.output_dir}")
