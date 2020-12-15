"""### Define TTS function"""

def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, style_wav=None,
                                                                             truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
    mel_postnet_spec = ap._denormalize(mel_postnet_spec)
    print(mel_postnet_spec.shape)
    print("max- ", mel_postnet_spec.max(), " -- min- ", mel_postnet_spec.min())
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(ap_vocoder._normalize(mel_postnet_spec).T).unsqueeze(0), hop_size=ap_vocoder.hop_length)
    if use_cuda:
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    print(waveform.shape)
    print(" >  Run-time: {}".format(time.time() - t_1))
    if figures:  
        visualize(alignment, mel_postnet_spec, stop_tokens, text, ap.hop_length, CONFIG, ap._denormalize(mel_spec))                                                                       
    
    OUT_FOLDER = 'result'
    os.makedirs(OUT_FOLDER, exist_ok=True)
    file_name = text.replace(" ", "_").replace(".","") + ".wav"

    ap.save_wav(waveform, file_name)
    
    return alignment, mel_postnet_spec, stop_tokens, waveform

"""### Load Models"""

import os
import torch
import yaml
import time
import librosa
import librosa.display

from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis
from TTS.utils.visual import visualize

from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.utils.audio import AudioProcessor as AudioProcessorVocoder

# model paths
TTS_MODEL = "tts_model/checkpoint_670000.pth.tar"
TTS_CONFIG = "tts_model/config.json"
PWGAN_MODEL = "pwgan_model/checkpoint-400000steps.pkl"
PWGAN_CONFIG = "pwgan_model/config.yml"

# load TTS config
TTS_CONFIG = load_config(TTS_CONFIG)

# load PWGAN config
with open(PWGAN_CONFIG) as f:
    PWGAN_CONFIG = yaml.load(f, Loader=yaml.Loader)
    
# Run FLAGs
use_cuda = False
# Set some config fields manually for testing
TTS_CONFIG.windowing = True
TTS_CONFIG.use_forward_attn = True 
# Set the vocoder
use_gl = False # use GL if True
batched_wavernn = True    # use batched wavernn inference if True

# LOAD TTS MODEL
# multi speaker 
speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load the audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)         

# load model state
cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()
print(cp['step'])
print(cp['r'])

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

# load PWGAN
if use_gl == False:
    vocoder_model = ParallelWaveGANGenerator(**PWGAN_CONFIG["generator_params"])
    vocoder_model.load_state_dict(torch.load(PWGAN_MODEL, map_location="cpu")["model"]["generator"])
    vocoder_model.remove_weight_norm()
    ap_vocoder = AudioProcessorVocoder(**PWGAN_CONFIG['audio'])    
    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval()

sentence = "Hello, sir!"
align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, use_gl=use_gl, figures=True)