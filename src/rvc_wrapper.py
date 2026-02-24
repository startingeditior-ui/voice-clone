import os
import sys
import torch
import numpy as np
import soundfile as sf

# Resolve project root and RVC directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVC_ROOT = os.path.join(PROJECT_ROOT, "Retrieval-based-Voice-Conversion-WebUI")
sys.path.append(RVC_ROOT)

from configs.config import Config
from infer.modules.vc.modules import VC

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
        project_root = os.getcwd()
        os.chdir(RVC_ROOT)
        try:
            self.vc.get_vc(model_name)
        finally:
            os.chdir(project_root)
        self.current_model = model_name

    def convert(self, source_path, output_path, f0_up_key=0, f0_method="rmvpe",
                index_rate=0.75, filter_radius=3, rms_mix_rate=0.25, protect=0.33):
        """
        Convert source audio using the loaded target voice.

        Args:
            source_path: Path to the input audio file
            output_path: Path to save the converted audio
            f0_up_key: Pitch shift in semitones
            f0_method: Pitch extraction method ('pm', 'harvest', 'crepe', 'rmvpe')
            index_rate: Index rate for feature retrieval (0.0 - 1.0). Higher = more like model voice.
            filter_radius: Median filter for pitch smoothing (1-7). Higher = smoother.
            rms_mix_rate: Volume envelope mix (0.0 - 1.0). Lower = keep original dynamics.
            protect: Consonant/breath protection (0.0 - 0.5). Lower = more protection.
        """
        if not self.current_model:
            raise ValueError("No model loaded. Call load_model() first.")

        # Ensure paths are absolute before moving to RVC_ROOT
        source_path = os.path.abspath(source_path)
        output_path = os.path.abspath(output_path)

        print(f"Converting {source_path} using {self.current_model}...")

        project_root = os.getcwd()
        os.chdir(RVC_ROOT)

        try:
            info, (tgt_sr, audio_opt) = self.vc.vc_single(
                0,           # sid
                source_path,
                f0_up_key,
                None,        # f0_file
                f0_method,
                "",          # file_index path
                "",          # file_index2
                index_rate,
                filter_radius,
                0,           # resample_sr (0 = no resample)
                rms_mix_rate,
                protect
            )

            if "Success" in info and audio_opt is not None:
                os.chdir(project_root)
                sf.write(output_path, audio_opt, tgt_sr)
                print(f"Conversion successful: {output_path}")
                return output_path
            else:
                raise Exception(f"RVC Conversion failed: {info}")
        finally:
            if os.getcwd() != project_root:
                os.chdir(project_root)
