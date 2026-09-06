"""NVIDIA TAO ReIdentificationNet (ResNet50, deployable_v1.2).

Runs on the best available accelerator:
  CUDA (ORT or PyTorch) -> MPS (PyTorch) -> CoreML (ORT) -> CPU

ONNX Runtime has no MPS provider. On Apple Silicon we convert the ONNX
graph to PyTorch and execute on torch.device("mps").
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NGC_ONNX_URL = (
    "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/"
    "reidentificationnet/deployable_v1.2/files?redirect=true"
    "&path=resnet50_market1501_aicity156.onnx"
)

PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
PIXEL_STD = np.array([0.226, 0.226, 0.226], dtype=np.float32).reshape(1, 3, 1, 1)
INPUT_H, INPUT_W = 256, 128


def default_model_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "models" / "resnet50_market1501_aicity156.onnx"


def ensure_model(path: Path | None = None) -> Path:
    path = Path(path) if path else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading ReIdentificationNet ONNX to {path} ...")
    urllib.request.urlretrieve(NGC_ONNX_URL, path)
    return path


def _torch_devices():
    try:
        import torch
    except ImportError:
        return False, False
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return cuda, mps


def pick_runtime() -> tuple[str, str]:
    """Return (backend, device_label). backend is 'torch' or 'ort'."""
    ort_eps = ort.get_available_providers()
    cuda, mps = _torch_devices()
    if cuda and "CUDAExecutionProvider" in ort_eps:
        return "ort", "CUDA"
    if cuda:
        return "torch", "CUDA"
    if mps:
        return "torch", "MPS"
    if "CoreMLExecutionProvider" in ort_eps:
        return "ort", "CoreML"
    return "ort", "CPU"


class ReIDEmbedder:
    def __init__(self, model_path: Path | None = None) -> None:
        model_path = ensure_model(model_path)
        self.backend, self.device_name = pick_runtime()
        self._torch_model = None
        self._torch_device = None
        self.session = None
        self.input_name = "input"
        self.output_name = "fc_pred"

        if self.backend == "torch":
            self._init_torch(model_path)
        else:
            self._init_ort(model_path)
        print(f"ReID backend: {self.backend}  device: {self.device_name}")

    def _init_ort(self, model_path: Path) -> None:
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.device_name == "CUDA" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:  # CoreML/CUDA can fail to compile the graph
            print(f"{providers[0]} unavailable ({exc.__class__.__name__}); falling back to CPU.")
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        used = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        if "CUDA" in used:
            self.device_name = "CUDA"
        elif "CoreML" in used:
            self.device_name = "CoreML"
        else:
            self.device_name = "CPU"

    def _init_torch(self, model_path: Path) -> None:
        import torch
        from onnx2torch import convert

        device = torch.device("cuda" if self.device_name == "CUDA" else "mps")
        model = convert(str(model_path))
        model.eval()
        model.to(device)
        self._torch_model = model
        self._torch_device = device

    def preprocess_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        batch = np.zeros((len(crops_bgr), 3, INPUT_H, INPUT_W), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            x = resized.astype(np.float32) / 255.0
            batch[i] = np.transpose(x, (2, 0, 1))
        batch = (batch - PIXEL_MEAN) / PIXEL_STD
        return batch

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 256), dtype=np.float32)
        tensor = self.preprocess_crops(crops_bgr)
        if self._torch_model is not None:
            import torch

            with torch.no_grad():
                x = torch.from_numpy(tensor).to(self._torch_device)
                feats = self._torch_model(x)
                feats = feats.detach().float().cpu().numpy()
        else:
            feats = self.session.run([self.output_name], {self.input_name: tensor})[0]
            feats = np.asarray(feats, dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return feats / norms
