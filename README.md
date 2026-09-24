# MotionSalt-Upscaler

Anime / video upscaling in Google Colab with a one-click Gradio UI — **2× super-resolution** with quality presets, per-pass detail/noise/halo/sharpen controls, and multiple pro encoders (x264, x265 8/10-bit, ProRes 4444 12-bit, FFV1 16-bit lossless).

## Credit & origin

**This project is a rebuild/fork of "Proteus V3" (V15.5), an open-source Google Colab upscaler released publicly by its original creator.** All credit for the underlying model, inference pipeline, tiling strategy, temporal processing, and encoder recipes goes to the original Proteus V3 author (identifiable from the released files as Hugging Face user [`legend2008`](https://huggingface.co/legend2008), whose public model bucket hosts the original TensorRT engines).

MotionSalt's contribution is limited to:

- a new **Gradio `Blocks` web interface** (the original shipped as bare Colab form fields),
- **permanent re-hosting** of the model weights as GitHub Release assets (the original relied on Dropbox / HF bucket links surviving),
- removal of a hardcoded Hugging Face token from the original setup cell (it was only a pip-cache speed optimization; no token of any kind is required or present here),
- this documentation and packaging.

> ⚖️ **License note:** the original released files contain **no explicit license**. They were published publicly as "OpenSource Codes" with public, tokenless weight links, so this fork proceeds in good faith under the **MIT License**, which covers MotionSalt's added code. Rights to the original Proteus V3 model and method remain with the original creator; if you are the creator and want anything changed or removed, open an issue and it will be addressed promptly.

## Usage

1. Open [`MotionSalt_Upscaler.ipynb`](MotionSalt_Upscaler.ipynb) in Google Colab (GPU runtime: *Runtime → Change runtime type → GPU*).
2. Run the single code cell. It installs dependencies, downloads the model weights from this repo's [Releases](../../releases), and launches the UI.
3. The cell prints both an inline UI and a public `*.gradio.live` link — the public link opens the same interface full-screen in a new tab while it keeps using the Colab GPU in the background.
4. Upload a video **or** paste a direct video URL, adjust parameters, press **Run**.
5. When finished you get: an in-UI video preview, a direct download button, and a temporary share link (auto-uploaded to a free temp host; expiry shown in the UI).
6. Inputs and outputs are always written under `/content/` (`/content/input_uploads`, `/content/outputs`) so they remain recoverable from Colab's file browser even if the Gradio tab is closed.

## Parameters

| Parameter | Options / Range | Default | Notes |
|---|---|---|---|
| Scale | `1x`, `2x` | `2x` | 1x enhances without changing size |
| Quality Preset | `Custom`, `V1 Anime 1 Pass`, `V2 Anime 2 Pass`, `V3 Anime High Quality 2 Pass` | `Custom` | Presets override the sliders |
| Speed (precision) | `ON` (FP16), `OFF` (FP32) | `ON` | FP16 faster; FP32 max precision |
| Anti-Alias & Deblur | −100 … 100 | 50 | Clears blur and jagged edges |
| Reduce Noise | 0 … 100 | 17 | Removes grain/artifacts |
| Recover Details | 0 … 100 | 100 | Restores fine detail |
| Dehalo | 0 … 100 | 10 | Removes white edge outlines |
| Sharpen | 0 … 100 | 20 | Edge sharpness |
| Revert Compression | 0 … 100 | 100 | Fixes compression artifacts |
| Recover Original Details | 0 … 100 | 0 | Blends original high-freq detail back |
| Input Video Resolution | `1080p`, `1440p`, `2160p`, `No Resize` | `1080p` | Pre-resize before AI |
| Input Image Resolution | `1080p`, `1440p`, `2160p`, `No Resize` | `1440p` | Pre-resize for images |
| Output Video Resize | `Original AI Output`, `1080p HD`, `1440p 2K`, `2160p 4K` | `1440p 2K` | Post-resize after AI |
| Image Format | `PNG 8bit`, `PNG 16bit`, `TIFF 10/12/16bit` | `PNG 8bit` | For image inputs |
| Encoder | `x264 8bit`, `x265 8bit`, `x265 10bit`, `ProRes 4444 12bit`, `FFV1 16bit Lossless` | `x265 8bit` | FFV1/ProRes are high-bit-depth/lossless |
| Save Frames | `ON`, `OFF` | `OFF` | Save every frame as 16-bit PNG |
| Quality | 0 … 51 | 16 | Lower = higher quality (13 = lossless) |

## Model weights

Mirrored as GitHub Release assets (tag `v15.5-weights`) so the project no longer depends on the original Dropbox/HF links:

- `prob-v3-fgnet-fp16-576x672-1x.trt` / `-2x.trt` — TensorRT FP16 engines (fast path)
- `prob-v3-fgnet-fp32-576x672-1x.trt` / `-2x.trt` — TensorRT FP32 engines (precision path)
- `prob-v3-fgnet-fp32-576x672-2x.onnx` — ONNX FP32 (CPU/portable fallback)

## License

MIT — see [LICENSE](LICENSE). Applies to MotionSalt's additions; the original Proteus V3 code/model remain the original creator's work (no explicit license was provided with the open-source release).
