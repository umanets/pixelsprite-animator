# Third-party software and models

The code of this repository is under the MIT licence (`LICENSE`). **The model weights are not**: the app downloads
them on its first start, and they keep their own, non-commercial licences.

## Models (downloaded on the first start, not part of this repository or of the release zip)

| Model | Source | Licence |
|---|---|---|
| pixelsprite canvas v2 (text → sprite) | `ihor-umanets/pixelsprite-canvas-v2` on Hugging Face | CC BY-NC 4.0 |
| AnimGPT (sprite → animation) | `ihor-umanets/pixelsprite-animgpt` on Hugging Face | CC BY-NC 4.0; trained on renders made with Qwen-Image 2.1, so the Qwen Research License terms apply too ("Built with Qwen"). See the model card. |
| Text encoder | `Xenova/paraphrase-multilingual-MiniLM-L12-v2` (ONNX build of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) | Apache-2.0 (base model) |
| Caption-mixing lexicon and example captions | `github.com/umanets/pixelsprite`, `docs/demo` | Apache-2.0 |

Sprites and animations you make with these weights are for non-commercial use.

## Libraries in the release zip

| File | What | Licence |
|---|---|---|
| `PixelSpriteStudio.exe` | This app; ONNX Runtime 1.31 (pyke's build) is linked in, with the Rust crates listed in `Cargo.lock` (egui / eframe, ort, tokenizers, image, gif, zip, serde, sha2, anyhow, …) | MIT (app, ONNX Runtime); MIT or Apache-2.0 (crates) |
| `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` | ONNX Runtime execution providers | MIT |
| `DirectML.dll` | Microsoft DirectML | MIT |
| `msvcp140*.dll`, `vcruntime140*.dll` | Microsoft Visual C++ runtime, app-local copy | Microsoft Visual Studio redistributable terms |
| Geist and Geist Mono fonts (embedded in the exe) | Vercel | SIL Open Font License 1.1 (`fonts/OFL.txt`) |

## NVIDIA libraries (optional, downloaded only when you press "Download CUDA libraries")

The CUDA 13 runtime, cuBLAS, cuFFT and cuDNN 9 are fetched from NVIDIA's own packages on PyPI
(`nvidia-cuda-runtime`, `nvidia-cublas`, `nvidia-cufft`, `nvidia-cudnn-cu13`) and stay on your machine under
NVIDIA's licence terms (each package carries its `License.txt`). They are not redistributed here.
