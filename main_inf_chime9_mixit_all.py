import argparse, os
import numpy
import soundfile
from tools import *
from trainer_chime9_mixit import *
from dataLoader_chime9_mixit_inference import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser(description = "Audio-visual target speaker extraction inference for session 0 all speakers.")

parser.add_argument('--batch_size', type=int,   default=1,        help='Batch size for inference')
parser.add_argument('--n_cpu',      type=int,   default=0,        help='Number of loader threads')
parser.add_argument('--init_model', type=str,   default="",       help='Path to model file')
parser.add_argument('--data_list',  type=str,   default="",       help='The path of the data list')
parser.add_argument('--backbone',   type=str,   default="seanet", help='Model backbone (seanet)')
parser.add_argument('--session_id', type=str,   default="session_00", help='Session ID to infer')
parser.add_argument('--track',      type=str,   default="track_00", help='Track to use')
parser.add_argument('--output_dir', type=str,   default="",       help='Output directory for separated audio files')
parser.add_argument('--save_path',  type=str,   default="",       help='Path for save_path (used by init_system)')
# trainer初期化に必要な引数を追加（推論時は使用されないが、初期化に必要）
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
inf_loader_obj = inf_loader_session_speakers(args.data_list, session_id=args.session_id, track=args.track)
args.infLoader = torch.utils.data.DataLoader(
	inf_loader_obj,
	batch_size=args.batch_size,
	shuffle=False,
	num_workers=args.n_cpu,
	drop_last=False
)

# Create output directory
if args.output_dir == "":
	args.output_dir = os.path.join(args.save_path, 'inference', args.session_id)
os.makedirs(args.output_dir, exist_ok=True)

# Save mixture audio (input audio)
mixture_audio_original = inf_loader_obj.mixture_audio_original
mixture_output_path = os.path.join(args.output_dir, f"mixture_{args.track}_input.wav")
soundfile.write(mixture_output_path, mixture_audio_original, 16000)
print(f"Saved mixture (input) audio to: {mixture_output_path}")

# Run inference
print(f"Starting inference for {args.session_id}...")
print(f"Output directory: {args.output_dir}")
print(f"Number of speakers: {len(args.infLoader)}")

s.eval()
B = args.batch_size
import time
time_start = time.time()

separated_audios = []
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
	
	spk_id = others['spk_id'][0] if isinstance(others['spk_id'], list) else others['spk_id']
	scale = others['scale'][0] if isinstance(others['scale'], list) else others['scale']
	
	output_filename = f"{spk_id}_{args.track}_separated.wav"
	output_path = os.path.join(args.output_dir, output_filename)
	
	output_audio = numpy.multiply(out_cat[0].cpu().numpy(), scale)
	soundfile.write(output_path, output_audio, 16000)
	
	separated_audios.append(output_audio)
	separated_audio_dict[spk_id] = output_audio
	
	time_used = time.time() - time_start
	print(f"Processed {num}/{len(args.infLoader)}: {spk_id} -> {output_path} (time: {time_used:.2f}s)")

# Sum all separated audios
print("\nSumming all separated audios...")
if len(separated_audios) > 0:
	min_separated_length = min([len(audio) for audio in separated_audios])
	mixture_length = len(mixture_audio_original)
	min_length = min(min_separated_length, mixture_length)
	
	truncated_audios = []
	for audio in separated_audios:
		if hasattr(audio, 'numpy'):
			audio = audio.numpy()
		elif hasattr(audio, 'cpu'):
			audio = audio.cpu().numpy()
		truncated_audios.append(numpy.asarray(audio, dtype=numpy.float32)[:min_length])
	
	summed_audio = numpy.sum(numpy.array(truncated_audios), axis=0)
	mixture_truncated = numpy.asarray(mixture_audio_original[:min_length], dtype=numpy.float32)
	
	summed_output_path = os.path.join(args.output_dir, f"summed_{args.track}_separated.wav")
	soundfile.write(summed_output_path, summed_audio, 16000)
	print(f"Saved summed separated audio to: {summed_output_path}")
	
	print("\nCalculating SI-SNR metrics...")
	
	mixture_tensor = torch.FloatTensor(mixture_truncated).unsqueeze(0)
	summed_tensor = torch.FloatTensor(summed_audio).unsqueeze(0)
	
	sisnr_summed = cal_SISNR(mixture_tensor, summed_tensor)
	sisnr_summed_value = sisnr_summed.item()
	
	results_path = os.path.join(args.output_dir, "results.txt")
	with open(results_path, 'w') as f:
		f.write("=" * 60 + "\n")
		f.write("SI-SNR Evaluation Results\n")
		f.write("=" * 60 + "\n")
		f.write(f"Session: {args.session_id}\n")
		f.write(f"Track: {args.track}\n")
		f.write(f"Number of speakers: {len(separated_audios)}\n")
		f.write(f"Audio length: {min_length} samples ({min_length/16000:.2f} seconds)\n")
		f.write("\n")
		f.write("-" * 60 + "\n")
		f.write("Metrics:\n")
		f.write("-" * 60 + "\n")
		f.write(f"SI-SNR (Summed Separated vs Original Mixture): {sisnr_summed_value:.4f} dB\n")
		f.write("\n")
		f.write("-" * 60 + "\n")
		f.write("Output Files:\n")
		f.write("-" * 60 + "\n")
		f.write(f"Mixture (input): {os.path.basename(mixture_output_path)}\n")
		f.write(f"Summed separated: {os.path.basename(summed_output_path)}\n")
		for spk_id in sorted(separated_audio_dict.keys()):
			f.write(f"{spk_id}: {spk_id}_{args.track}_separated.wav\n")
	
	print(f"\nResults saved to: {results_path}")
	print(f"SI-SNR (Summed Separated vs Original Mixture): {sisnr_summed_value:.4f} dB")
else:
	print("\nWarning: No separated audios to sum!")

print(f"\nInference completed! Output files saved to: {args.output_dir}")
print(f"Mixture (input) audio saved to: {mixture_output_path}")
if len(separated_audios) > 0:
	print(f"Summed separated audio saved to: {summed_output_path}")

