import os
import sys
import torch
import numpy as np
import soundfile as sf

# Add RVC directory to path
RVC_ROOT = os.path.join(os.path.dirname(__file__), "Retrieval-based-Voice-Conversion-WebUI")
sys.path.append(RVC_ROOT)

from configs.config import Config
from infer.modules.vc.modules import VC

import torch

# Monkeypatch torch.load for compatibility with PyTorch 2.4+ (WeightsUnpickler error)
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load

class RVCWrapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", is_half=None):
        os.environ["weight_root"] = os.path.join(RVC_ROOT, "assets", "weights")
        os.environ["weight_uvr5_root"] = os.path.join(RVC_ROOT, "assets", "uvr5_weights")
        os.environ["index_root"] = os.path.join(RVC_ROOT, "logs")
        os.environ["outside_index_root"] = os.path.join(RVC_ROOT, "assets", "indices")
        os.environ["rmvpe_root"] = os.path.join(RVC_ROOT, "assets", "rmvpe")
        
        # Save project root and change to RVC root for initialization
        project_root = os.getcwd()
        os.chdir(RVC_ROOT)
        try:
            self.config = Config()
            self.config.device = device
            self.config.is_half = is_half if is_half is not None else (device == "cuda")
            self.vc = VC(self.config)
        finally:
            os.chdir(project_root)
        self.current_model = None

    def load_model(self, model_name):
        """Load an RVC model (.pth file)"""
        if self.current_model == model_name:
            return
        
        print(f"Loading RVC model: {model_name}")
        # sid is the filename in assets/weights
        # Change to RVC root for model loading
        project_root = os.getcwd()
        os.chdir(RVC_ROOT)
        try:
            self.vc.get_vc(model_name)
        finally:
            os.chdir(project_root)
        self.current_model = model_name

    def convert(self, source_path, output_path, f0_up_key=0, f0_method="rmvpe", index_rate=0.75):
        """
        Convert source audio using the loaded target voice.
        f0_up_key: Pitch shift (semitones)
        f0_method: 'pm', 'harvest', 'crepe', 'rmvpe'
        """
        if not self.current_model:
            raise ValueError("No model loaded. Call load_model() first.")

        # Ensure paths are absolute before moving to RVC_ROOT
        source_path = os.path.abspath(source_path)
        output_path = os.path.abspath(output_path)

        print(f"Converting {source_path} using {self.current_model}...")
        
        # Save project root and change to RVC root for conversion
        project_root = os.getcwd()
        os.chdir(RVC_ROOT)
        
        try:
            # vc_single arguments:
            # sid, input_audio_path, f0_up_key, f0_file, f0_method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect
            info, (tgt_sr, audio_opt) = self.vc.vc_single(
                0, # sid (not used for weight loading anymore as get_vc handled it)
                source_path,
                f0_up_key,
                None, # f0_file
                f0_method,
                "", # file_index (path)
                "", # file_index2 (dropdown choice, usually index is auto-detected)
                index_rate,
                3, # filter_radius
                0, # resample_sr (0 = no resample)
                0.25, # rms_mix_rate
                0.33 # protect
            )

            if "Success" in info and audio_opt is not None:
                # Switch back before writing file
                os.chdir(project_root)
                sf.write(output_path, audio_opt, tgt_sr)
                print(f"Conversion successful: {output_path}")
                return output_path
            else:
                raise Exception(f"RVC Conversion failed: {info}")
        finally:
            if os.getcwd() != project_root:
                os.chdir(project_root)

# Example usage (commented out):
# rvc = RVCWrapper()
# rvc.load_model("my_voice_model.pth")
# rvc.convert("input.wav", "output.wav")
