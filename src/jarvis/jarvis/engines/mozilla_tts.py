# MIT License

# Copyright (c) 2019 Georgios Papachristou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import threading
import logging
import pyttsx3
import queue
import numpy as np
import simpleaudio as sa

from jarvis.core.console import ConsoleManager

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
TTS_MODEL = "tts_model2/checkpoint_670000.pth.tar"
TTS_CONFIG = "tts_model2/config.json"
PWGAN_MODEL = "pwgan_model2/checkpoint-400000steps.pkl"
PWGAN_CONFIG = "pwgan_model2/config.yml"

# load TTS config
TTS_CONFIG = load_config(TTS_CONFIG)

# load PWGAN config
with open(PWGAN_CONFIG) as f:
    PWGAN_CONFIG = yaml.load(f, Loader=yaml.Loader)

class MozillaTTSMLModelClient:
    def __init__(self, model, CONFIG, use_cuda, ap, use_gl, speaker_id, vocoder_model, ap_vocoder):
        self.model = model
        self.CONFIG = CONFIG
        self.use_cuda = use_cuda
        self.ap = ap
        self.use_gl = use_gl
        self.speaker_id = speaker_id
        self.vocoder_model = vocoder_model
        self.ap_vocoder = ap_vocoder

    def say(self, text: str):
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens = synthesis(self.model, text, self.CONFIG, self.use_cuda, self.ap, self.speaker_id, style_wav=None,
                                                                                truncated=False, enable_eos_bos_chars=self.CONFIG.enable_eos_bos_chars)
        if self.CONFIG.model == "Tacotron" and not self.use_gl:
            mel_postnet_spec = self.ap.out_linear_to_mel(mel_postnet_spec.T).T
        mel_postnet_spec = self.ap._denormalize(mel_postnet_spec)
        print(mel_postnet_spec.shape)
        print("max- ", mel_postnet_spec.max(), " -- min- ", mel_postnet_spec.min())
        if not self.use_gl:
            waveform = self.vocoder_model.inference(torch.FloatTensor(self.ap_vocoder._normalize(mel_postnet_spec).T).unsqueeze(0), hop_size=self.ap_vocoder.hop_length)
        if self.use_cuda:
            waveform = waveform.cpu()
        waveform = waveform.numpy()                                                            

        wav_norm = waveform * (32767 / max(0.01, np.max(np.abs(waveform))))
        # Start playback
        play_obj = sa.play_buffer(wav_norm.astype(np.int16), 1, 2, self.CONFIG["audio"]["sample_rate"])

        # Wait for playback to finish before exiting
        play_obj.wait_done()
        
        return alignment, mel_postnet_spec, stop_tokens, waveform
    
    
class MozillaTTS:
    """
    Text To Speech Engine (TTS)
    """

    def __init__(self):
        self.tts_engine = self._set_voice_engine()

    def run_engine(self):
        pass

    @staticmethod
    def _set_voice_engine():
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
        num_chars = len(phonemes) if TTS_CONFIG["use_phonemes"] else len(symbols)
        model = setup_model(num_chars, len(speakers), TTS_CONFIG)

        # load the audio processor
        ap = AudioProcessor(**TTS_CONFIG["audio"])         

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

        return MozillaTTSMLModelClient(model, TTS_CONFIG, use_cuda, ap, use_gl, speaker_id, vocoder_model, ap_vocoder)


class MozillaTTSEngine(MozillaTTS):
    def __init__(self):
        super().__init__()
        self.logger = logging
        self.message_queue = queue.Queue(maxsize=9)  # Maxsize is the size of the queue / capacity of messages
        self.stop_speaking = False
        self.console_manager = ConsoleManager()

    def assistant_response(self, message, refresh_console=True):
        """
        Assistant response in voice.
        :param refresh_console: boolean
        :param message: string
        """
        self._insert_into_message_queue(message)
        try:
            self._speech_and_console(refresh_console)
        except RuntimeError as e:
            self.logger.error('Error in assistant response thread with message {0}'.format(e))

    def _insert_into_message_queue(self, message):
        try:
            self.message_queue.put(message)
        except Exception as e:
            self.logger.error("Unable to insert message to queue with error message: {0}".format(e))

    def _speech_and_console(self, refresh_console):
        """
        Speech method translate text batches to speech and print them in the console.
        :param text: string (e.g 'tell me about google')
        """
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get()
                if message:
                    self.tts_engine.say(message)
                    self.console_manager.console_output(message, refresh_console=refresh_console)
                    self.run_engine()
        except Exception as e:
            self.logger.error("Speech and console error message: {0}".format(e))
