"""
proteus_core.py — MotionSalt Upscaler inference core.

Adapted from the open-source "Proteus V3" V15.5 Cell 2 by its original
creator (HF user: legend2008). Same model, tiling, temporal-state and
encoder recipes; refactored into a class so a UI can call it
programmatically. All credit for the method belongs to the original author.
"""
import os, time, math, subprocess, shutil, json, threading, sys
import numpy as np
import cv2

try:
    import tensorrt as trt
    _HAS_TRT = True
except ImportError:
    trt = None
    _HAS_TRT = False

try:
    import pycuda.driver as cuda
    _HAS_PYCUDA = True
except ImportError:
    cuda = None
    _HAS_PYCUDA = False


# ── CUDA context management ──────────────────────────────────────────────────
# Root cause of "explicit_context_dependent failed: invalid device context - no
# currently active context?": this module previously imported pycuda.autoinit,
# which binds the CUDA context to the IMPORTING thread. Gradio runs inference on
# a worker thread, so every TensorRT call there executed with NO current context.
# Fix: lazily create + push the device primary context on whichever thread
# actually runs TensorRT (never autoinit at import time).
_CUDA_CTX_HOLDER = {}


def ensure_cuda_context(thread_key=None):
    """Initialise CUDA and make a context current on the CALLING thread.
    Returns True on success, False if CUDA/pycuda is unavailable."""
    if not _HAS_PYCUDA:
        return False
    key = thread_key if thread_key is not None else threading.get_ident()
    if key in _CUDA_CTX_HOLDER:
        return True
    try:
        cuda.init()
        try:
            if cuda.Context.get_current() is not None:
                _CUDA_CTX_HOLDER[key] = None  # context owned elsewhere; still fine
                return True
        except Exception:
            pass
        ctx = cuda.Device(0).retain_primary_context()
        ctx.push()
        _CUDA_CTX_HOLDER[key] = ctx
        return True
    except Exception:
        return False


def gpu_summary():
    """Human-readable identity of the CURRENT GPU (for logs/diagnostics)."""
    if not _HAS_PYCUDA:
        return "unknown (pycuda missing)"
    try:
        cuda.init()
        dev = cuda.Device(0)
        cc = dev.compute_capability()
        return f"{dev.name()} (CC {cc[0]}.{cc[1]})"
    except Exception as e:
        return f"unknown ({e})"


class MotionSaltUpscaler:
    IMAGE_PASSES = 3
    VIDEO_FIRST_PASSES = 3
    PREFLIGHT = 10
    TILE_H, TILE_W = 576, 672
    OVERLAP = 192
    CHANNELS = {1: 15, 2: 24}
    SWS_FLAGS = 'spline+accurate_rnd+full_chroma_int'
    COLOR_TRC, COLORSPACE, COLOR_PRIMARIES = '1', '5', '6'
    MOVFLAGS = 'frag_keyframe+empty_moov+delay_moov+use_metadata_tags+write_colr'
    USE_TEMPORAL = True

    QUALITY_PRESETS = {
        "V1 Anime 1 Pass": {
            "scale": 2, "passes": 1,
            "anti_alias_deblur": 100, "reduce_noise": 100, "recover_details": 100,
            "dehalo": 50, "sharpen": 35, "revert_compression": 100, "recover_original": 0,
        },
        "V2 Anime 2 Pass": {
            "scale": 2, "passes": 2,
            "pass1": {"anti_alias_deblur": -50, "reduce_noise": 100, "recover_details": 79,
                      "dehalo": 65, "sharpen": 5, "revert_compression": 100, "recover_original": 0},
            "pass2": {"anti_alias_deblur": 80, "reduce_noise": 100, "recover_details": 87,
                      "dehalo": 70, "sharpen": 20, "revert_compression": 100, "recover_original": 0},
        },
        "V3 Anime High Quality 2 Pass": {
            "scale": 2, "passes": 2,
            "pass1": {"anti_alias_deblur": 100, "reduce_noise": 100, "recover_details": 100,
                      "dehalo": 45, "sharpen": 20, "revert_compression": 100, "recover_original": 0},
            "pass2": {"anti_alias_deblur": 100, "reduce_noise": 100, "recover_details": 100,
                      "dehalo": 50, "sharpen": 23, "revert_compression": 100, "recover_original": 0},
        },
    }

    _INPUT_RES_MAP = {"1080p": 1080, "1440p": 1440, "2160p": 2160, "No Resize": 99999}
    _OUTPUT_RES_MAP = {"Original AI Output": 0, "1080p HD": 1080, "1440p 2K": 1440, "2160p 4K": 2160}

    def __init__(self, input_path, scale=2, quality_preset="Custom", speed_up="ON",
                 anti_alias_deblur=50, reduce_noise=17, recover_details=100, dehalo=10,
                 sharpen=20, revert_compression=100, recover_original=0,
                 input_video_resolution="1080p", input_image_resolution="1440p",
                 output_video_resize="1440p 2K", image_format="PNG 8bit",
                 encoder="x265 8bit", save_frames="OFF", quality=16,
                 multipass="OFF", custom_pass2=None,
                 verbose=True, log=print):
        self.log = log
        self.verbose = verbose
        self.INPUT_PATH = input_path
        self.SCALE = int(str(scale).replace("x", ""))
        self.Quality_Preset = quality_preset
        self.Upscaling_Speed_Up = speed_up
        self.Anti_Alias_Deblur = int(anti_alias_deblur)
        self.ReduceNoise = int(reduce_noise)
        self.RecoverDetails = int(recover_details)
        self.Dehalo = int(dehalo)
        self.Sharpen = int(sharpen)
        self.RevertCompression = int(revert_compression)
        self.Recover_Original_Details = int(recover_original)
        self.Input_Video_Resolution = input_video_resolution
        self.Input_Image_Resolution = input_image_resolution
        self.Output_Video_Resize = output_video_resize
        self.Image_Format = image_format
        self.Encoder = encoder
        self.Save_Frames = save_frames
        self.Quality = int(quality)

        self.MAX_INPUT_HEIGHT = self._INPUT_RES_MAP.get(input_video_resolution, 1080)
        self.MAX_IMAGE_INPUT_DIM = self._INPUT_RES_MAP.get(input_image_resolution, 2160)
        self.OUTPUT_TARGET_SHORT_SIDE = self._OUTPUT_RES_MAP.get(output_video_resize, 0)
        self.IS_IMAGE_INPUT = any(input_path.lower().endswith(e)
                                  for e in ['.jpg', '.png', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'])

        # Preset / slider resolution (identical semantics to original Cell 2)
        self.MULTIPASS = (multipass is True) or (str(multipass).strip().upper()
                          in ("ON", "TRUE", "1", "YES"))
        if quality_preset != "Custom" and quality_preset in self.QUALITY_PRESETS:
            _preset = self.QUALITY_PRESETS[quality_preset]
            self.SCALE = _preset["scale"]
            self.PASS_COUNT = _preset["passes"]
            self.MULTIPASS = self.PASS_COUNT > 1
            if self.PASS_COUNT == 1:
                self.PASS1_SLIDERS = {
                    "anti_alias_deblur": _preset["anti_alias_deblur"],
                    "reduce_noise": _preset["reduce_noise"],
                    "recover_details": _preset["recover_details"],
                    "dehalo": _preset["dehalo"],
                    "sharpen": _preset["sharpen"],
                    "revert_compression": _preset["revert_compression"],
                    "recover_original": _preset["recover_original"],
                }
                self.PASS2_SLIDERS = None
            else:
                self.PASS1_SLIDERS = dict(_preset["pass1"])
                self.PASS2_SLIDERS = dict(_preset["pass2"])
        else:
            # Custom: multipass is user-controlled and drives the exact same
            # two-pass structure the built-in 2-pass presets use.
            self.PASS_COUNT = 2 if self.MULTIPASS else 1
            self.PASS1_SLIDERS = {
                "anti_alias_deblur": self.Anti_Alias_Deblur,
                "reduce_noise": self.ReduceNoise,
                "recover_details": self.RecoverDetails,
                "dehalo": self.Dehalo,
                "sharpen": self.Sharpen,
                "revert_compression": self.RevertCompression,
                "recover_original": self.Recover_Original_Details,
            }
            if self.PASS_COUNT == 2:
                # Pass 2 defaults to the same slider values unless the caller
                # supplies explicit pass-2 overrides (preset-style parity).
                _p2 = dict(self.PASS1_SLIDERS)
                if custom_pass2:
                    _p2.update({k: int(v) for k, v in custom_pass2.items()
                                if v is not None})
                self.PASS2_SLIDERS = _p2
            else:
                self.PASS2_SLIDERS = None

        self.INJECT_OVERLAY = {}
        self.preBlur_val = self.noise_val = self.details_val = 0.0
        self.halo_val = self.sharpen_val = self.compression_val = 0.0
        self.CURRENT_RECOVER_ORIGINAL = 0
        self.set_pass_params(self.PASS1_SLIDERS)

        self._USE_FP16 = (speed_up == "ON")
        self._PRECISION_TAG = "fp16" if self._USE_FP16 else "fp32"
        self._NP_IO_DTYPE = np.float32
        self.prev_lr = None
        self.prev_hr = None
        self._load_model()

        self.FULL_FEATHER_MASK = self.make_feather_mask(
            self.TILE_H * self.SCALE, self.TILE_W * self.SCALE, self.OVERLAP * self.SCALE)

    # ------------------------------------------------------------------ model
    def _load_model(self):
        SCALE = self.SCALE
        _CANDIDATE_PATHS = [
            f'/content/models/prob-v3-fgnet-{self._PRECISION_TAG}-576x672-{SCALE}x.trt',
            f'/content/prob-v3-fgnet-{self._PRECISION_TAG}-576x672-{SCALE}x.trt',
            f'/content/models/prob-v3-fgnet-{"fp32" if self._USE_FP16 else "fp16"}-576x672-{SCALE}x.trt',
            f'/content/prob-v3-fgnet-{"fp32" if self._USE_FP16 else "fp16"}-576x672-{SCALE}x.trt',
            f'/content/models/prob-v3-fgnet-fp32-576x672-{SCALE}x.onnx',
            f'/content/prob-v3-fgnet-fp32-576x672-{SCALE}x.onnx',
        ]
        self.MODEL_PATH = None
        self.MODEL_IS_ONNX = False
        self._USE_ORT = False
        for _p in _CANDIDATE_PATHS:
            if os.path.exists(_p):
                self.MODEL_PATH = _p
                self.MODEL_IS_ONNX = _p.endswith('.onnx')
                break
        if self.MODEL_PATH is None:
            raise FileNotFoundError(f"Model file not found for SCALE={SCALE}x under /content/models")

        try:
            import torch as _torch_check
            _HAS_CUDA = _torch_check.cuda.is_available()
        except Exception:
            _HAS_CUDA = False
        if not _HAS_CUDA and not self.MODEL_IS_ONNX:
            for _p in _CANDIDATE_PATHS:
                if _p.endswith('.onnx') and os.path.exists(_p):
                    self.MODEL_PATH = _p
                    self.MODEL_IS_ONNX = True
                    break

        self.log(f"Loading model: {os.path.basename(self.MODEL_PATH)}")
        self.log(f"Runtime GPU: {gpu_summary()}")

        if self.MODEL_IS_ONNX:
            self._init_onnx(_HAS_CUDA)
            return

        # ── TensorRT path (fast, but NOT portable) ───────────────────────────
        # A .trt engine is compiled for ONE GPU architecture / driver / TRT
        # version. Colab assigns different GPUs per session (T4, L4, A100…), so
        # the engine must be validated against the CURRENT GPU. Strategy: make a
        # CUDA context current on THIS thread, deserialize, then run a real
        # smoke inference on zeros. ANY failure -> automatic ONNX fallback.
        _onnx_path = next((_p for _p in _CANDIDATE_PATHS
                           if _p.endswith('.onnx') and os.path.exists(_p)), None)
        try:
            if not _HAS_TRT:
                raise RuntimeError("tensorrt python package not importable")
            if not _HAS_CUDA:
                raise RuntimeError("no CUDA GPU available in this session")
            if not ensure_cuda_context():
                raise RuntimeError("could not create a current CUDA context on this thread")
            self._init_trt_engine()
            self.log(f"TRT engine loaded + smoke-tested on {gpu_summary()}. "
                     f"Tile {self.TILE_H}x{self.TILE_W}, scale {self.SCALE}x, "
                     f"{os.path.getsize(self.MODEL_PATH)/1048576:.1f} MB")
            return
        except Exception as _trt_err:
            self.log(f"WARNING: TensorRT path unusable on this GPU/session: {_trt_err}")
            if _onnx_path is None:
                raise
            self.log(f"-> Auto-fallback to portable ONNX model: {os.path.basename(_onnx_path)}")
            self.MODEL_PATH = _onnx_path
            self.MODEL_IS_ONNX = True
            self._init_onnx(_HAS_CUDA)

    def _init_onnx(self, has_cuda):
        import onnxruntime as _ort
        _providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                      if has_cuda else ['CPUExecutionProvider'])
        self.ort_sess = _ort.InferenceSession(self.MODEL_PATH, providers=_providers)
        _active = self.ort_sess.get_providers()
        _input_meta = self.ort_sess.get_inputs()[0]
        self.input_name = _input_meta.name
        self.output_name = self.ort_sess.get_outputs()[0].name
        input_shape = tuple(_input_meta.shape)
        self.output_shape = tuple(self.ort_sess.get_outputs()[0].shape)
        self.TILE_H, self.TILE_W = input_shape[1], input_shape[2]
        self._USE_ORT = True
        self.d_input = self.d_output = self.stream = self.context = self.engine = None
        self.log(f"ONNX session ready (providers: {', '.join(_active)}). "
                 f"Tile {self.TILE_H}x{self.TILE_W}")

    def _init_trt_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.MODEL_PATH, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("engine failed to deserialize (built for a different GPU/TRT version)")
        self.context = self.engine.create_execution_context()
        input_shape = output_shape = None
        _io_dtype = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name, input_shape, _io_dtype = name, tuple(shape), dtype
            else:
                self.output_name, output_shape = name, tuple(shape)
        self.output_shape = output_shape
        self.TILE_H, self.TILE_W = input_shape[1], input_shape[2]
        _actual_ch = input_shape[-1]
        _detected_scale = {15: 1, 24: 2}.get(_actual_ch, self.SCALE)
        if _detected_scale != self.SCALE:
            self.log(f"Engine is {_detected_scale}x but SCALE={self.SCALE}x — auto-correcting.")
        self.SCALE = _detected_scale
        self._NP_IO_DTYPE = np.float16 if _io_dtype == trt.DataType.HALF else np.float32
        _IO_ITEMSIZE = np.dtype(self._NP_IO_DTYPE).itemsize
        self.d_input = cuda.mem_alloc(trt.volume(input_shape) * _IO_ITEMSIZE)
        self.d_output = cuda.mem_alloc(trt.volume(output_shape) * _IO_ITEMSIZE)
        self.stream = cuda.Stream()
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        # Smoke-run one real inference on zeros: proves context+engine+GPU work
        # together NOW (at load time) instead of failing mid-video.
        _smoke_in = np.zeros(tuple(input_shape), dtype=self._NP_IO_DTYPE)
        cuda.memcpy_htod_async(self.d_input, _smoke_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        _smoke_out = np.empty(self.output_shape, dtype=self._NP_IO_DTYPE)
        cuda.memcpy_dtoh_async(_smoke_out, self.d_output, self.stream)
        self.stream.synchronize()
        if not np.isfinite(_smoke_out).all():
            raise RuntimeError("smoke inference produced non-finite output")

    # ------------------------------------------------------------------ params
    def set_pass_params(self, slider_dict):
        self.INJECT_OVERLAY = {
            3: float(slider_dict["anti_alias_deblur"]) / 100.0,
            4: float(slider_dict["reduce_noise"]) / 100.0,
            5: float(slider_dict["recover_details"]) / 100.0,
            6: float(slider_dict["dehalo"]) / 100.0,
            7: float(slider_dict["sharpen"]) / 100.0,
            8: float(slider_dict["revert_compression"]) / 100.0,
        }
        self.preBlur_val = self.INJECT_OVERLAY[3]
        self.noise_val = self.INJECT_OVERLAY[4]
        self.details_val = self.INJECT_OVERLAY[5]
        self.halo_val = self.INJECT_OVERLAY[6]
        self.sharpen_val = self.INJECT_OVERLAY[7]
        self.compression_val = self.INJECT_OVERLAY[8]
        self.CURRENT_RECOVER_ORIGINAL = int(slider_dict.get("recover_original", 0))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def make_feather_mask(h, w, overlap):
        m = np.ones((h, w), dtype=np.float32)
        if overlap > 0:
            r = np.linspace(0, 1, overlap, dtype=np.float32)
            m[:, :overlap] *= r[np.newaxis, :]
            m[:, -overlap:] *= r[::-1][np.newaxis, :]
            m[:overlap, :] *= r[:, np.newaxis]
            m[-overlap:, :] *= r[::-1][:, np.newaxis]
        return m

    @staticmethod
    def reflect_pad(img, th, tw):
        h, w = img.shape[:2]
        ph, pw = max(0, th - h), max(0, tw - w)
        if ph > 0 or pw > 0:
            return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        return img

    @staticmethod
    def compute_tile_positions(ts, tile, overlap):
        if ts <= tile:
            return [0], 1, tile
        nb = math.ceil((ts - overlap) / (tile - overlap))
        if nb <= 1:
            return [0], 1, tile
        step = (ts - tile) / (nb - 1)
        pos = [int(round(i * step)) for i in range(nb)]
        pos[-1] = ts - tile
        return pos, nb, ts

    @staticmethod
    def pixel_shuffle_down(hr_img, scale):
        h, w, c = hr_img.shape
        new_h, new_w = h // scale, w // scale
        img = hr_img.reshape(new_h, scale, new_w, scale, c)
        img = img.transpose(0, 2, 1, 3, 4)
        img = img.reshape(new_h, new_w, c * scale * scale)
        return np.ascontiguousarray(img)

    def build_input(self, lr_curr, params, lr_prev, packed_hr_prev):
        channels = [lr_curr, params, lr_prev, packed_hr_prev]
        _dtype = self._NP_IO_DTYPE
        inp = np.ascontiguousarray(np.concatenate(channels, axis=-1), dtype=_dtype)
        if self.INJECT_OVERLAY:
            for ch_idx, val in self.INJECT_OVERLAY.items():
                inp[:, :, ch_idx] = val
        inp = inp[None]
        assert inp.shape[-1] == self.CHANNELS[self.SCALE], \
            f"Expected {self.CHANNELS[self.SCALE]}ch, got {inp.shape[-1]}"
        return inp

    @staticmethod
    def preprocess_frame(rgb_f32, preblur_val):
        return np.clip(rgb_f32, 0, 1).astype(np.float32)

    def get_media_info(self, path):
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif')
        if path.lower().endswith(img_exts):
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(path) as img:
                    w, h = img.size
                return w, h, 1.0, 1
            except Exception as e:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h, 1.0, 1
                raise RuntimeError(f"Failed to read image dimensions for {path}: {e}")
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration:stream_tags=rotate:format=duration',
                            '-of', 'json', path], capture_output=True, text=True, timeout=30)
        try:
            data = json.loads(r.stdout)
            if 'streams' not in data or len(data['streams']) == 0:
                raise RuntimeError(f"ffprobe: no video streams found in {path}")
            info = data['streams'][0]
            w, h = int(info['width']), int(info['height'])
            rotate = 0
            if 'tags' in info and 'rotate' in info['tags']:
                try:
                    rotate = int(float(info['tags']['rotate']))
                except ValueError:
                    rotate = 0
            if rotate in (90, 270, -90, -270):
                w, h = h, w
                self.log(f"Detected {rotate} deg rotation. Display: {w}x{h}")
            num, den = info['r_frame_rate'].split('/')
            fps = float(num) / float(den) if float(den) > 0 else 25.0
            tf_str = info.get('nb_frames', '0')
            tf = int(tf_str) if tf_str and tf_str.isdigit() and int(tf_str) > 0 else None
            if tf is None:
                # Some containers (notably Matroska — the Pass 1 FFV1 .mkv
                # intermediate) store NO per-stream frame count, so nb_frames
                # comes back empty where the original upload (mp4 etc.) had one.
                # Fallback: estimate total frames from container duration x fps.
                # Matroska stores a reliable segment duration, and the processing
                # loop self-corrects tf if the estimate is off, so this is safe.
                # (A full ffprobe -count_frames scan would be exact but requires
                # decoding the entire 16-bit FFV1 intermediate — minutes of I/O
                # for a progress-bar total.)
                dur = None
                for cand in (info.get('duration'),
                             (data.get('format') or {}).get('duration')):
                    try:
                        if cand is not None and float(cand) > 0:
                            dur = float(cand)
                            break
                    except (TypeError, ValueError):
                        continue
                if dur is not None and fps > 0:
                    tf = max(1, int(round(dur * fps)))
                    self.log(f"No stored frame count in this container; estimated "
                             f"frames={tf} from duration {dur:.3f}s x {fps:.2f}fps "
                             f"(total self-corrects during processing)")
                else:
                    self.log(f"WARNING: frame count unavailable for {path} and no "
                             f"duration to estimate from — progress will run without "
                             f"a total (processing itself is unaffected)")
            return w, h, fps, tf
        except (json.JSONDecodeError, KeyError, RuntimeError) as e:
            raise RuntimeError(f"ffprobe failed for {path}: {e}\nstderr: {r.stderr}")

    @staticmethod
    def read_image_16bit(path):
        from PIL import Image as _PILImage
        img = _PILImage.open(path)
        mode = img.mode
        if mode in ('I', 'I;16', 'I;16B', 'I;16L'):
            img = img.convert('I;16')
            arr = np.array(img, dtype=np.uint16)
            arr = np.stack([arr, arr, arr], axis=-1)
            input_bit_depth = 16
        elif mode in ('RGB', 'RGBA', 'P', 'L', 'CMYK'):
            if img.mode == 'RGB':
                try:
                    if img.filename and img.filename.lower().endswith(('.png', '.tiff', '.tif')):
                        img_raw = _PILImage.open(path)
                        extrema = img_raw.getextrema()
                        max_val = max(e[1] if isinstance(e, tuple) else e for e in extrema)
                        if max_val > 255:
                            arr_16 = np.array(img_raw, dtype=np.uint16)
                            if arr_16.ndim == 2:
                                arr_16 = np.stack([arr_16, arr_16, arr_16], axis=-1)
                            return arr_16, 16
                except Exception:
                    pass
            img_rgb = img.convert('RGB')
            arr_8 = np.array(img_rgb, dtype=np.uint8)
            arr_16 = (arr_8.astype(np.uint16) << 8) | arr_8.astype(np.uint16)
            input_bit_depth = 8
        else:
            img_rgb = img.convert('RGB')
            arr_8 = np.array(img_rgb, dtype=np.uint8)
            arr_16 = (arr_8.astype(np.uint16) << 8) | arr_8.astype(np.uint16)
            input_bit_depth = 8
        return arr_16, input_bit_depth

    @staticmethod
    def compute_resize_dimensions(iw, ih, target_short_side):
        if target_short_side <= 0:
            return iw, ih
        short_side = min(iw, ih)
        if short_side <= target_short_side:
            return iw, ih
        scale = target_short_side / short_side
        new_w = int(round(iw * scale))
        new_h = int(round(ih * scale))
        new_w -= new_w % 2
        new_h -= new_h % 2
        return new_w, new_h

    @staticmethod
    def detect_orientation(w, h):
        if w > h:
            return "landscape"
        elif h > w:
            return "portrait"
        return "square"

    @staticmethod
    def resize_frame_16bit(frame_uint16, target_w, target_h):
        h, w = frame_uint16.shape[:2]
        if (w, h) == (target_w, target_h):
            return frame_uint16
        return cv2.resize(frame_uint16, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def apply_recover_original_detail(self, model_hr, lr_curr, scale, blend=0):
        if blend == 0:
            return model_hr
        alpha = blend / 100.0
        h, w = lr_curr.shape[:2]
        original_hr = cv2.resize(lr_curr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        detail = original_hr - cv2.GaussianBlur(original_hr, (5, 5), 1.0)
        return np.clip(model_hr + alpha * detail, 0.0, 1.0)

    # ------------------------------------------------------------------ inference
    def run_model_tiles(self, lr_curr, params, lr_prev, packed_hr, tile_h, tile_w, overlap):
        SCALE = self.SCALE
        in_h, in_w = lr_curr.shape[:2]
        out_h, out_w = in_h * SCALE, in_w * SCALE
        yp, nh, ph = self.compute_tile_positions(in_h, tile_h, overlap)
        xp, nw, pw = self.compute_tile_positions(in_w, tile_w, overlap)
        lr_curr_p = self.reflect_pad(lr_curr, ph, pw)
        params_p = self.reflect_pad(params, ph, pw)
        lr_prev_p = self.reflect_pad(lr_prev, ph, pw)
        packed_hr_p = self.reflect_pad(packed_hr, ph, pw)
        poh, pow_ = ph * SCALE, pw * SCALE
        obuf = np.zeros((poh, pow_, 3), dtype=np.float32)
        wbuf = np.zeros((poh, pow_), dtype=np.float32)
        for ty in range(nh):
            for tx in range(nw):
                y0, x0 = yp[ty], xp[tx]
                t_curr = lr_curr_p[y0:y0 + tile_h, x0:x0 + tile_w]
                t_params = params_p[y0:y0 + tile_h, x0:x0 + tile_w]
                t_prev = lr_prev_p[y0:y0 + tile_h, x0:x0 + tile_w]
                t_hr = packed_hr_p[y0:y0 + tile_h, x0:x0 + tile_w]
                inp = self.build_input(t_curr, t_params, t_prev, t_hr)
                if self._USE_ORT:
                    out_tile = self.ort_sess.run(None, {self.input_name: inp})[0][0]
                else:
                    cuda.memcpy_htod_async(self.d_input, inp, self.stream)
                    self.context.execute_async_v3(self.stream.handle)
                    out_tile = np.empty(self.output_shape, dtype=self._NP_IO_DTYPE)
                    cuda.memcpy_dtoh_async(out_tile, self.d_output, self.stream)
                    self.stream.synchronize()
                    out_tile = out_tile[0].astype(np.float32)
                out_tile = np.clip(out_tile, 0, 1)
                oy0, ox0 = y0 * SCALE, x0 * SCALE
                oe, xe = oy0 + tile_h * SCALE, ox0 + tile_w * SCALE
                for c in range(3):
                    obuf[oy0:oe, ox0:xe, c] += out_tile[:, :, c] * self.FULL_FEATHER_MASK
                wbuf[oy0:oe, ox0:xe] += self.FULL_FEATHER_MASK
        wbuf = np.maximum(wbuf, 1e-6)
        return np.clip(obuf / wbuf[:, :, np.newaxis], 0, 1)[:out_h, :out_w]

    def process_frame(self, input_rgb48, is_warmup=False):
        in_h, in_w = input_rgb48.shape[:2]
        rgb_f32 = input_rgb48.astype(np.float32) / 65535.0
        lr_curr = self.preprocess_frame(rgb_f32, self.preBlur_val)
        H, W, _ = lr_curr.shape
        params = np.concatenate([
            np.full((H, W, 1), self.preBlur_val, dtype=np.float32),
            np.full((H, W, 1), self.noise_val, dtype=np.float32),
            np.full((H, W, 1), self.details_val, dtype=np.float32),
            np.full((H, W, 1), self.halo_val, dtype=np.float32),
            np.full((H, W, 1), self.sharpen_val, dtype=np.float32),
            np.full((H, W, 1), self.compression_val, dtype=np.float32),
        ], axis=-1)
        if self.prev_lr is None:
            lr_prev = lr_curr.copy() if self.USE_TEMPORAL else np.zeros_like(lr_curr)
            hr_simple = cv2.resize(lr_curr, (in_w * self.SCALE, in_h * self.SCALE),
                                   interpolation=cv2.INTER_CUBIC)
            packed_hr = self.pixel_shuffle_down(hr_simple, self.SCALE)
            n_passes = self.IMAGE_PASSES if self.IS_IMAGE_INPUT else self.VIDEO_FIRST_PASSES
            for p in range(n_passes):
                hr_pass = self.run_model_tiles(lr_curr, params, lr_prev, packed_hr,
                                               self.TILE_H, self.TILE_W, self.OVERLAP)
                hr_pass = np.clip(hr_pass, 0, 1)
                if p < n_passes - 1:
                    packed_hr = self.pixel_shuffle_down(hr_pass, self.SCALE)
        else:
            lr_prev = self.prev_lr if self.USE_TEMPORAL else np.zeros_like(lr_curr)
            packed_hr = self.pixel_shuffle_down(self.prev_hr, self.SCALE) if self.USE_TEMPORAL else \
                self.pixel_shuffle_down(cv2.resize(lr_curr, (in_w * self.SCALE, in_h * self.SCALE),
                                                   interpolation=cv2.INTER_CUBIC), self.SCALE)
        hr_output = self.run_model_tiles(lr_curr, params, lr_prev, packed_hr,
                                         self.TILE_H, self.TILE_W, self.OVERLAP)
        hr_output = np.clip(hr_output, 0, 1)
        if self.CURRENT_RECOVER_ORIGINAL > 0:
            hr_output = self.apply_recover_original_detail(hr_output, lr_curr, self.SCALE,
                                                           self.CURRENT_RECOVER_ORIGINAL)
            hr_output = np.clip(hr_output, 0, 1)
        if not is_warmup:
            self.prev_lr = lr_curr.copy()
            self.prev_hr = hr_output.copy()
        return np.ascontiguousarray((hr_output * 65535).astype(np.uint16))

    # ------------------------------------------------------------------ ffmpeg I/O
    class FFmpegRGB48Reader:
        def __init__(self, path, orig_w, orig_h, out_w, out_h, fps=25.0, seek_frame=0):
            self.w, self.h = out_w, out_h
            self.frame_bytes = out_w * out_h * 6
            cmd = ['ffmpeg', '-v', 'error']
            if seek_frame > 0:
                cmd.extend(['-ss', f'{seek_frame / fps:.6f}'])
            cmd.extend(['-i', path])
            if (orig_w, orig_h) != (out_w, out_h):
                cmd.extend(['-vf', f'scale={out_w}:{out_h}:flags=lanczos'])
            cmd.extend(['-sws_flags', MotionSaltUpscaler.SWS_FLAGS, '-pix_fmt', 'rgb48le',
                        '-vsync', 'passthrough', '-f', 'rawvideo', 'pipe:1'])
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._stderr_lines = []
            def _drain():
                for line in self.proc.stderr:
                    self._stderr_lines.append(line)
            threading.Thread(target=_drain, daemon=True).start()

        def read(self):
            data = b''
            while len(data) < self.frame_bytes:
                chunk = self.proc.stdout.read(self.frame_bytes - len(data))
                if not chunk:
                    break
                data += chunk
            if len(data) < self.frame_bytes:
                return None
            return np.frombuffer(data, dtype=np.uint16).reshape((self.h, self.w, 3)).copy()

        def close(self):
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def setup_encoder(self, out_w, out_h, fps, out_temp_path, encoder, quality, iha, input_path):
        SWS_FLAGS, COLOR_TRC, COLORSPACE, COLOR_PRIMARIES, MOVFLAGS = (
            self.SWS_FLAGS, self.COLOR_TRC, self.COLORSPACE, self.COLOR_PRIMARIES, self.MOVFLAGS)
        is_x264_8bit = (encoder == "x264 8bit")
        is_x265_8bit = (encoder == "x265 8bit")
        is_x265_10bit = (encoder == "x265 10bit")
        is_prores_12 = (encoder == "ProRes 4444 12bit")
        is_ffv1_16 = (encoder == "FFV1 16bit Lossless")
        video_codec = profile_v = pix_fmt_out = tag_v = input_pix_fmt = vf_filter = None
        if is_x264_8bit or is_x265_8bit:
            video_codec = "h264_nvenc" if is_x264_8bit else "hevc_nvenc"
            profile_v = "high" if is_x264_8bit else "main"
            pix_fmt_out, input_pix_fmt, vf_filter = "yuv420p", "bgr24", ""
            tag_v = "avc1" if is_x264_8bit else "hvc1"
        elif is_x265_10bit:
            video_codec, profile_v, pix_fmt_out, tag_v = "hevc_nvenc", "main10", "p010le", "hvc1"
            input_pix_fmt, vf_filter = "bgr48le", "format=p010le"
        elif is_prores_12:
            bits_per_mb = max(100, min(800, int(800 - (quality / 51.0) * 700)))
            video_codec, profile_v, pix_fmt_out, input_pix_fmt = "prores_ks", "4", "yuv444p12le", "rgb48le"
            tag_v = ""
            vf_filter = "scale=in_range=full:in_color_matrix=bt709:out_range=tv:out_color_matrix=bt709"
        elif is_ffv1_16:
            video_codec, profile_v, pix_fmt_out, input_pix_fmt, tag_v, vf_filter = \
                "ffv1", "", "rgb48le", "rgb48le", "", ""
        else:
            raise RuntimeError(f"Unknown encoder: {encoder}")

        enc_cmd = ['ffmpeg', '-y', '-thread_queue_size', '512',
                   '-f', 'rawvideo', '-pix_fmt', input_pix_fmt,
                   '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', '-']
        if iha:
            enc_cmd.extend(['-thread_queue_size', '512', '-i', input_path])
        enc_cmd.extend(['-map', '0:v:0'])
        if iha:
            enc_cmd.extend(['-map', '1:a:0'])
        if is_ffv1_16:
            enc_cmd.extend(['-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1',
                            '-g', '300', '-slices', '16', '-slicecrc', '1', '-pix_fmt', 'rgb48le'])
            if iha:
                enc_cmd.extend(['-c:a', 'flac'])
            enc_cmd.append(out_temp_path)
        elif is_prores_12:
            enc_cmd.extend(['-vf', vf_filter, '-sws_flags', SWS_FLAGS,
                            '-color_trc', COLOR_TRC, '-colorspace', COLORSPACE,
                            '-color_primaries', COLOR_PRIMARIES, '-color_range', 'tv',
                            '-c:v', video_codec, '-profile:v', profile_v, '-pix_fmt', pix_fmt_out,
                            '-bits_per_mb', str(bits_per_mb)])
            if iha:
                enc_cmd.extend(['-c:a', 'pcm_s16le'])
            enc_cmd.extend(['-movflags', MOVFLAGS, out_temp_path])
        else:
            if vf_filter:
                enc_cmd.extend(['-vf', vf_filter])
            enc_cmd.extend(['-sws_flags', SWS_FLAGS, '-color_trc', COLOR_TRC,
                            '-colorspace', COLORSPACE, '-color_primaries', COLOR_PRIMARIES,
                            '-c:v', video_codec, '-profile:v', profile_v, '-pix_fmt', pix_fmt_out,
                            '-preset', 'p7', '-tune', 'hq', '-rc', 'constqp', '-qp', str(quality),
                            '-multipass', '2', '-spatial-aq', '1', '-temporal-aq', '1',
                            '-rc-lookahead', '60', '-tag:v', tag_v])
            if iha:
                enc_cmd.extend(['-c:a', 'copy'])
            enc_cmd.extend(['-movflags', MOVFLAGS, out_temp_path])

        enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        enc_errors = []
        def _read_enc_stderr():
            for line in enc_proc.stderr:
                enc_errors.append(line.decode('utf-8', errors='ignore'))
        threading.Thread(target=_read_enc_stderr, daemon=True).start()
        time.sleep(2)
        if enc_proc.poll() is not None:
            raise RuntimeError(f"Encoder failed to start!\nFFmpeg stderr:\n{''.join(enc_errors[-20:])}")
        enc_info = {'is_x264_8bit': is_x264_8bit, 'is_x265_8bit': is_x265_8bit,
                    'is_x265_10bit': is_x265_10bit, 'is_prores_12': is_prores_12,
                    'is_ffv1_16': is_ffv1_16}
        return enc_proc, enc_errors, enc_info

    @staticmethod
    def write_frame_to_encoder(enc_proc, enc_info, result_uint16_rgb):
        if enc_info['is_x264_8bit'] or enc_info['is_x265_8bit']:
            result_8bit = cv2.cvtColor((result_uint16_rgb >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR)
            enc_proc.stdin.write(result_8bit.tobytes())
        elif enc_info['is_x265_10bit']:
            enc_proc.stdin.write(np.ascontiguousarray(
                cv2.cvtColor(result_uint16_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint16).tobytes())
        else:
            enc_proc.stdin.write(np.ascontiguousarray(result_uint16_rgb, dtype=np.uint16).tobytes())
        enc_proc.stdin.flush()

    @staticmethod
    def has_audio(path):
        try:
            r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a',
                                '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', path],
                               capture_output=True, text=True, timeout=10)
            return 'audio' in r.stdout
        except Exception:
            return False

    # ------------------------------------------------------------------ main video pass
    def process_video_pass(self, input_path, output_path, encoder, quality,
                           apply_input_resize=True, preflight=True, pass_name="Pass",
                           audio_source_path=None, output_resize=True, is_final_pass=True,
                           progress_cb=None):
        if audio_source_path is None:
            audio_source_path = input_path
        iw, ih, fps, tf = self.get_media_info(input_path)
        self.log(f"\n===== {pass_name} =====")
        self.log(f"Input: {input_path}  {iw}x{ih}  {fps:.2f}fps  frames={tf}")
        pw, ph = self.compute_resize_dimensions(iw, ih, self.MAX_INPUT_HEIGHT) if apply_input_resize else (iw, ih)
        self.log(f"Input Resize: {iw}x{ih} -> {pw}x{ph}")
        ai_out_w, ai_out_h = pw * self.SCALE, ph * self.SCALE
        if output_resize and self.OUTPUT_TARGET_SHORT_SIDE > 0:
            out_w, out_h = self.compute_resize_dimensions(ai_out_w, ai_out_h, self.OUTPUT_TARGET_SHORT_SIDE)
        else:
            out_w, out_h = ai_out_w, ai_out_h
        self.log(f"Pipeline: {pw}x{ph} -> AI {self.SCALE}x -> {out_w}x{out_h}")
        iha = self.has_audio(audio_source_path)

        FD = '/content/upscaled_frames'
        use_save_frames = (self.Save_Frames == "ON") and is_final_pass
        if use_save_frames:
            shutil.rmtree(FD, ignore_errors=True)
            os.makedirs(FD)

        enc_proc = enc_errors = enc_info = None
        if not use_save_frames:
            enc_proc, enc_errors, enc_info = self.setup_encoder(
                out_w, out_h, fps, output_path, encoder, quality, iha, audio_source_path)

        if preflight and self.PREFLIGHT > 0:
            self.log("Engine warmup...")
            pre_reader = self.FFmpegRGB48Reader(input_path, iw, ih, pw, ph, fps, 0)
            wf = []
            for _ in range(self.PREFLIGHT):
                f = pre_reader.read()
                if f is None:
                    break
                wf.append(f)
            pre_reader.close()
            _wbar = None
            try:
                from tqdm.auto import tqdm as _tqdm_warm
                _wbar = _tqdm_warm(total=len(wf[:self.PREFLIGHT]), desc=f"{pass_name} warmup",
                                   unit="frame", leave=False, ncols=100)
            except Exception:
                _wbar = None
            for w in wf[:self.PREFLIGHT]:
                self.process_frame(w, is_warmup=True)
                if _wbar is not None:
                    _wbar.update(1)
            if _wbar is not None:
                _wbar.close()
            self.prev_lr = None
            self.prev_hr = None

        reader = self.FFmpegRGB48Reader(input_path, iw, ih, pw, ph, fps, seek_frame=0)
        fi = 0
        st0 = time.time()
        stopped = False
        _pbar = None
        try:
            from tqdm.auto import tqdm as _tqdm_cls
            _pbar = _tqdm_cls(total=tf, desc=pass_name, unit="frame", ncols=100,
                              mininterval=0.5)
        except Exception:
            _pbar = None
        try:
            while True:
                frame = reader.read()
                if frame is None:
                    break
                fi += 1
                t0 = time.time()
                result_uint16_rgb = np.ascontiguousarray(self.process_frame(frame), dtype=np.uint16)
                if (out_w, out_h) != (ai_out_w, ai_out_h):
                    result_uint16_rgb = np.ascontiguousarray(
                        self.resize_frame_16bit(result_uint16_rgb, out_w, out_h), dtype=np.uint16)
                if use_save_frames:
                    cv2.imwrite(os.path.join(FD, f'f{fi:06d}.png'),
                                np.ascontiguousarray(cv2.cvtColor(result_uint16_rgb, cv2.COLOR_RGB2BGR),
                                                     dtype=np.uint16))
                else:
                    try:
                        self.write_frame_to_encoder(enc_proc, enc_info, result_uint16_rgb)
                    except BrokenPipeError:
                        raise RuntimeError(f"Pipe broken frame {fi}!\n{''.join(enc_errors[-30:])}")
                if tf is None or fi > tf:
                    tf = fi
                el = time.time() - st0
                fps2 = fi / el if el > 0 else 0
                eta = (tf - fi) / fps2 if tf and fps2 > 0 else 0
                msg = (f"{pass_name} frame {fi}/{tf} ({fi/tf*100:.1f}%) "
                       f"{time.time()-t0:.2f}s/f ETA {eta/60:.1f}m") if tf else \
                      f"{pass_name} frame {fi} {time.time()-t0:.2f}s/f"
                if progress_cb:
                    progress_cb(msg)
                if _pbar is not None:
                    # _pbar.total is None when the bar was created with tf=None
                    # (container without a stored frame count). tf gets backfilled
                    # to an int above, so guard on _pbar.total itself — never
                    # compare fi > None.
                    if _pbar.total is None or (tf and fi > _pbar.total):
                        _pbar.total = tf if tf else fi
                    _pbar.update(1)
                    _pbar.set_postfix_str(f"{time.time()-t0:.2f}s/f" + (f" ETA {eta/60:.1f}m" if tf else ""))
        finally:
            if _pbar is not None:
                _pbar.close()
            try:
                reader.close()
            except Exception:
                pass
            if enc_proc:
                for fn in (enc_proc.stdin.flush, enc_proc.stdin.close):
                    try:
                        fn()
                    except Exception:
                        pass
                try:
                    ret = enc_proc.wait(timeout=120)
                    if ret != 0:
                        self.log(f"Encoder warning (exit {ret}): {''.join(enc_errors[-10:])[-400:]}")
                except subprocess.TimeoutExpired:
                    enc_proc.kill()
                    enc_proc.wait()
        return fi, fps, tf, stopped, out_w, out_h

    def lossless_resize_video(self, input_path, output_path, target_resolution):
        iw, ih, fps, tf = self.get_media_info(input_path)
        target_short = self._INPUT_RES_MAP.get(target_resolution, 99999)
        new_w, new_h = self.compute_resize_dimensions(iw, ih, target_short)
        cmd = ['ffmpeg', '-y', '-i', input_path,
               '-vf', f'scale={new_w}:{new_h}:flags=lanczos',
               '-sws_flags', self.SWS_FLAGS,
               '-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1',
               '-g', '300', '-slices', '16', '-slicecrc', '1', '-pix_fmt', 'rgb48le',
               '-an', output_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(f"Lossless resize failed: {r.stderr[-400:]}")

    @staticmethod
    def get_output_extension(encoder):
        if encoder == "FFV1 16bit Lossless":
            return '.mkv'
        if encoder == "ProRes 4444 12bit":
            return '.mov'
        return '.mp4'

    def remux_audio(self, temp_path, final_path, audio_source_path):
        iha = self.has_audio(audio_source_path)
        cmd = ['ffmpeg', '-y', '-i', temp_path]
        if iha:
            cmd.extend(['-i', audio_source_path])
        cmd.extend(['-c:v', 'copy', '-map', '0:v'])
        if iha:
            cmd.extend(['-c:a', 'copy', '-map', '1:a:0'])
        cmd.extend(['-map_metadata', '0', '-movflags', 'use_metadata_tags', final_path])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            self.log(f"Audio remux warning: {r.stderr[-300:]}")

    def encode_from_saved_frames(self, FD, fi, fps, out_temp_path, encoder, quality, input_path):
        self.log(f"Encoding from saved frames ({fi} frames, {fps:.2f} fps)...")
        vd = fi / fps if fps > 0 else 0
        iha = self.has_audio(input_path)
        base = ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(FD, 'f%06d.png'),
                '-i', input_path, '-map', '0:v:0']
        if encoder == "FFV1 16bit Lossless":
            cmd = base + (['-map', '1:a:0', '-c:a', 'flac'] if iha else []) + \
                ['-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1',
                 '-g', '300', '-slices', '16', '-slicecrc', '1', '-pix_fmt', 'rgb48le',
                 '-movflags', self.MOVFLAGS, '-t', str(vd), '-map_metadata', '0', out_temp_path]
        elif encoder == "ProRes 4444 12bit":
            cmd = base + (['-map', '1:a:0', '-c:a', 'pcm_s16le'] if iha else []) + \
                ['-vf', 'scale=in_range=full:in_color_matrix=bt709:out_range=tv:out_color_matrix=bt709',
                 '-sws_flags', self.SWS_FLAGS, '-color_trc', self.COLOR_TRC,
                 '-colorspace', self.COLORSPACE, '-color_primaries', self.COLOR_PRIMARIES,
                 '-color_range', 'tv', '-c:v', 'prores_ks', '-profile:v', '4',
                 '-pix_fmt', 'yuv444p12le', '-bits_per_mb', '800',
                 '-movflags', self.MOVFLAGS, '-t', str(vd), '-map_metadata', '0', out_temp_path]
        else:
            vmap = {"x265 10bit": ("hevc_nvenc", "main10", "p010le", "hvc1", "format=p010le"),
                    "x265 8bit": ("hevc_nvenc", "main", "yuv420p", "hvc1", None),
                    "x264 8bit": ("h264_nvenc", "high", "yuv420p", "avc1", None)}
            if encoder not in vmap:
                raise RuntimeError(f"Unknown encoder for Save_Frames mode: {encoder}")
            codec, prof, pxf, tag, vf = vmap[encoder]
            cmd = base + ['-sws_flags', self.SWS_FLAGS, '-color_trc', self.COLOR_TRC,
                          '-colorspace', self.COLORSPACE, '-color_primaries', self.COLOR_PRIMARIES]
            if vf:
                cmd.extend(['-vf', vf])
            cmd += ['-c:v', codec, '-profile:v', prof, '-pix_fmt', pxf,
                    '-preset', 'p7', '-tune', 'hq', '-rc', 'constqp', '-qp', str(quality),
                    '-multipass', '2', '-spatial-aq', '1', '-temporal-aq', '1',
                    '-rc-lookahead', '60', '-tag:v', tag]
            if iha:
                cmd.extend(['-map', '1:a:0', '-c:a', 'copy'])
            cmd.extend(['-movflags', self.MOVFLAGS, '-t', str(vd), '-map_metadata', '0', out_temp_path])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg encode from saved frames failed: {r.stderr[-400:]}")

    # ------------------------------------------------------------------ entry point
    def run(self, output_dir="/content/outputs", progress_cb=None):
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(self.INPUT_PATH):
            raise FileNotFoundError(self.INPUT_PATH)
        bn = os.path.splitext(os.path.basename(self.INPUT_PATH))[0]

        if self.IS_IMAGE_INPUT:
            iw, ih, _, _ = self.get_media_info(self.INPUT_PATH)
            rw, rh = self.compute_resize_dimensions(iw, ih, self.MAX_IMAGE_INPUT_DIM)
            needs_downscale = (rw, rh) != (iw, ih)
            self.log(f"Image {iw}x{ih} -> {rw}x{rh}")
            frame, input_bit_depth = self.read_image_16bit(self.INPUT_PATH)
            if needs_downscale:
                frame = self.resize_frame_16bit(frame, rw, rh)
            if self.PASS_COUNT == 1:
                result_uint16_rgb = self.process_frame(frame)
            else:
                self.log("PASS 1 of 2 (image)")
                self.set_pass_params(self.PASS1_SLIDERS)
                self.prev_lr = None
                self.prev_hr = None
                pass1_result = self.process_frame(frame)
                target_h, target_w = frame.shape[:2]
                pass1_resized = np.ascontiguousarray(
                    self.resize_frame_16bit(pass1_result, target_w, target_h), dtype=np.uint16)
                self.log("PASS 2 of 2 (image)")
                self.set_pass_params(self.PASS2_SLIDERS)
                self.prev_lr = None
                self.prev_hr = None
                result_uint16_rgb = self.process_frame(pass1_resized)

            result_uint16_rgb = np.ascontiguousarray(result_uint16_rgb, dtype=np.uint16)
            img_bgr = np.ascontiguousarray(cv2.cvtColor(result_uint16_rgb, cv2.COLOR_RGB2BGR),
                                           dtype=np.uint16)
            OUT_BASE = os.path.join(output_dir, f"MotionSalt.{bn}_upscaled")
            if self.Image_Format == "PNG 8bit":
                out_path = OUT_BASE + '_8bit.png'
                cv2.imwrite(out_path, (img_bgr >> 8).astype(np.uint8))
            elif self.Image_Format == "PNG 16bit":
                out_path = OUT_BASE + '_16bit.png'
                cv2.imwrite(out_path, img_bgr)
            elif self.Image_Format == "TIFF 10bit":
                out_path = OUT_BASE + '_10bit.tiff'
                cv2.imwrite(out_path, np.left_shift(np.right_shift(img_bgr, 6), 6).astype(np.uint16))
            elif self.Image_Format == "TIFF 12bit":
                out_path = OUT_BASE + '_12bit.tiff'
                cv2.imwrite(out_path, np.left_shift(np.right_shift(img_bgr, 4), 4).astype(np.uint16))
            elif self.Image_Format == "TIFF 16bit":
                out_path = OUT_BASE + '_16bit.tiff'
                cv2.imwrite(out_path, img_bgr)
            else:
                out_path = OUT_BASE + '_16bit.png'
                cv2.imwrite(out_path, img_bgr)
            self.log(f"Image saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
            return out_path

        # ---- video ----
        OUT_EXT = self.get_output_extension(self.Encoder)
        OUT = os.path.join(output_dir, f"MotionSalt.{bn}_upscaled{OUT_EXT}")
        OUT_TEMP = os.path.join(output_dir, f"MotionSalt.{bn}_upscaled_temp{OUT_EXT}")
        if self.PASS_COUNT == 1:
            fi, fps, tf, stopped, out_w, out_h = self.process_video_pass(
                self.INPUT_PATH, OUT_TEMP, self.Encoder, self.Quality,
                apply_input_resize=True, preflight=True, pass_name="Processing",
                audio_source_path=self.INPUT_PATH, output_resize=True, is_final_pass=True,
                progress_cb=progress_cb)
            if self.Save_Frames == "ON":
                self.encode_from_saved_frames('/content/upscaled_frames', fi, fps, OUT_TEMP,
                                              self.Encoder, self.Quality, self.INPUT_PATH)
            self.remux_audio(OUT_TEMP, OUT, self.INPUT_PATH)
            if not os.path.exists(OUT) and os.path.exists(OUT_TEMP):
                shutil.move(OUT_TEMP, OUT)
            try:
                os.remove(OUT_TEMP)
            except Exception:
                pass
        else:
            temp_pass1 = os.path.join(output_dir, f"MotionSalt.{bn}_pass1.mkv")
            self.set_pass_params(self.PASS1_SLIDERS)
            self.prev_lr = None
            self.prev_hr = None
            self.log("PASS 1 of 2 — FFV1 16bit lossless intermediate")
            fi1, fps1, tf1, stopped1, _, _ = self.process_video_pass(
                self.INPUT_PATH, temp_pass1, "FFV1 16bit Lossless", 0,
                apply_input_resize=True, preflight=True, pass_name="Pass 1",
                audio_source_path=self.INPUT_PATH, output_resize=False, is_final_pass=False,
                progress_cb=progress_cb)
            if stopped1:
                self.remux_audio(temp_pass1, OUT, self.INPUT_PATH)
            else:
                temp_resized = os.path.join(output_dir, f"MotionSalt.{bn}_pass1_resized.mkv")
                self.lossless_resize_video(temp_pass1, temp_resized, self.Input_Video_Resolution)
                try:
                    os.remove(temp_pass1)
                except Exception:
                    pass
                self.set_pass_params(self.PASS2_SLIDERS)
                self.prev_lr = None
                self.prev_hr = None
                self.log(f"PASS 2 of 2 — encoder {self.Encoder}")
                fi2, fps2, tf2, stopped2, out_w, out_h = self.process_video_pass(
                    temp_resized, OUT_TEMP, self.Encoder, self.Quality,
                    apply_input_resize=False, preflight=True, pass_name="Pass 2",
                    audio_source_path=self.INPUT_PATH, output_resize=True, is_final_pass=True,
                    progress_cb=progress_cb)
                if self.Save_Frames == "ON":
                    self.encode_from_saved_frames('/content/upscaled_frames', fi2, fps2, OUT_TEMP,
                                                  self.Encoder, self.Quality, self.INPUT_PATH)
                try:
                    os.remove(temp_resized)
                except Exception:
                    pass
                self.remux_audio(OUT_TEMP, OUT, self.INPUT_PATH)
                if not os.path.exists(OUT) and os.path.exists(OUT_TEMP):
                    shutil.move(OUT_TEMP, OUT)
                try:
                    os.remove(OUT_TEMP)
                except Exception:
                    pass
        self.log(f"Done: {OUT} ({os.path.getsize(OUT)/1048576:.1f} MB)")
        return OUT
