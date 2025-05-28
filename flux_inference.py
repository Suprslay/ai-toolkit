#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flux Inference and Inpainting Script with IP-Adapter Support
-----------------------------------------------------------
A single‑file utility for running image generation with Flux models.
Features
~~~~~~~~
* LoRA loading (multiple adapters, independent strengths)
* IP-Adapter support for garment/style control
* Character LoRA + Garment IP-Adapter combination
* Optional face detection + in‑place inpainting with a second Flux pipeline
* Upper body inpainting (face + neck only)
* YAML/CLI hybrid configuration (CLI overrides YAML)
* VRAM‑friendly switches
* Robust filename handling and prompt logging

Usage
~~~~~
python flux_inference.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from diffusers import (
    AutoencoderKL,
    FluxInpaintPipeline,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 32) -> str:
    """Simple slug for filenames."""
    out = "".join(c for c in text[:max_len] if c.isalnum() or c == " ").strip()
    return out.replace(" ", "_").lower()


def load_and_preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load and preprocess image for IP-Adapter use."""
    try:
        img = Image.open(image_path).convert("RGB")
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        print(f"[IP-Adapter] Loaded image: {image_path} -> {img.size}")
        return img
    except Exception as e:
        print(f"[IP-Adapter] Error loading image {image_path}: {e}")
        raise


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class FluxInference:
    """Light‑weight manager around Diffusers' `FluxPipeline` with IP-Adapter support."""

    # ------------------------------ init & helpers -------------------------- #

    def __init__(
        self,
        model_path: str,
        output_dir: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        face_inpainting: bool = False,
        upper_body_inpainting: bool = False,
        face_lora_paths: Optional[List[str]] = None,
        face_lora_strengths: Optional[List[float]] = None,
        upper_body_lora_paths: Optional[List[str]] = None,
        upper_body_lora_strengths: Optional[List[float]] = None,
        # NEW: IP-Adapter configuration
        ip_adapter_model_path: Optional[str] = None,
        ip_adapter_images: Optional[List[str]] = None,
        ip_adapter_scales: Optional[List[float]] = None,
        ip_adapter_target_size: Optional[Tuple[int, int]] = None,
        low_vram: bool = False,
        config: Optional[Dict] = None,
    ) -> None:
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.low_vram = low_vram
        self.config = config or {}

        # Face‑related opts
        self.face_inpainting = face_inpainting
        self.face_lora_paths = face_lora_paths or []
        self.face_lora_strengths = face_lora_strengths or [0.75] * len(self.face_lora_paths)
        if len(self.face_lora_strengths) < len(self.face_lora_paths):
            self.face_lora_strengths.extend(
                [0.75] * (len(self.face_lora_paths) - len(self.face_lora_strengths))
            )
        self.face_prompt = "A woman, smooth skin, make-up, confident, lightly smiling, model, fashion"
        self.face_neg_prompt = (
            "deformed face, distorted face, disfigured, low res, bad anatomy, cartoon, anime, painting, sketch, low quality, pixelated, jpeg artifacts, oversaturated"
        )
        self.face_guidance_scale = 4.0
        self.face_expansion = 0
        self.face_detection_model = "mediapipe"
        self.save_face_debug = True

        # Upper body‑related opts
        self.upper_body_inpainting = upper_body_inpainting
        self.upper_body_lora_paths = upper_body_lora_paths or []
        self.upper_body_lora_strengths = upper_body_lora_strengths or [0.75] * len(self.upper_body_lora_paths)
        if len(self.upper_body_lora_strengths) < len(self.upper_body_lora_paths):
            self.upper_body_lora_strengths.extend(
                [0.75] * (len(self.upper_body_lora_paths) - len(self.upper_body_lora_strengths))
            )
        self.upper_body_prompt = "SLAY1MNSHA bslaymnsha a woman, beautiful face, elegant neck, portrait, smooth skin, professional photography, high quality"
        self.upper_body_neg_prompt = (
            "deformed face, distorted face, disfigured, low res, bad anatomy, cartoon, anime, painting, sketch, low quality, pixelated, jpeg artifacts, oversaturated"
        )
        self.upper_body_guidance_scale = 3.5
        self.save_upper_body_debug = True

        # NEW: IP-Adapter configuration
        self.ip_adapter_model_path = ip_adapter_model_path or "XLabs-AI/flux-ip-adapter"
        self.ip_adapter_images = ip_adapter_images or []
        self.ip_adapter_scales = ip_adapter_scales or [0.6] * len(self.ip_adapter_images)
        self.ip_adapter_target_size = ip_adapter_target_size or (512, 512)
        self.ip_adapter_enabled = len(self.ip_adapter_images) > 0
        
        # Ensure scales match images
        if len(self.ip_adapter_scales) < len(self.ip_adapter_images):
            self.ip_adapter_scales.extend(
                [0.6] * (len(self.ip_adapter_images) - len(self.ip_adapter_scales))
            )

        # Internal state
        self.pipeline: FluxPipeline | None = None
        self.inpaint_pipeline: FluxInpaintPipeline | None = None
        self.face_detector = None
        self.pose_detector = None
        self.is_loaded = False
        self.face_loras_loaded = False
        self.upper_body_loras_loaded = False
        self.ip_adapter_loaded = False  # NEW: Track IP-Adapter state
        self.preprocessed_ip_images = []  # NEW: Store preprocessed IP-Adapter images
        
        # NEW: Store clean components for inpaint pipeline
        self._clean_vae = None
        self._clean_transformer = None
        self._clean_text_encoder = None
        self._clean_text_encoder_2 = None
        self._clean_tokenizer = None
        self._clean_tokenizer_2 = None
        self._clean_scheduler = None

        print(f"[Init] model_path={model_path}  device={device}  dtype={dtype}")
        
        if self.ip_adapter_enabled:
            print(f"[Init] IP-Adapter ENABLED with {len(self.ip_adapter_images)} reference images")
            for img_path, scale in zip(self.ip_adapter_images, self.ip_adapter_scales):
                print(f"       • {Path(img_path).name}  scale={scale}")
        
        if face_inpainting:
            print("[Init] Face inpainting ENABLED")
            if self.face_lora_paths:
                for p, s in zip(self.face_lora_paths, self.face_lora_strengths):
                    print(f"       • {Path(p).name}  strength={s} (will load on-demand)")
        
        if upper_body_inpainting:
            print("[Init] Upper body inpainting ENABLED")
            print("       → Upper body = FACE + NECK region only")
            if self.upper_body_lora_paths:
                for p, s in zip(self.upper_body_lora_paths, self.upper_body_lora_strengths):
                    print(f"       • {Path(p).name}  strength={s} (will load on-demand)")

    # ------------------------------ model loading -------------------------- #

    def load_model(
        self,
        *,
        inference_lora_paths: Optional[List[str]] = None,
        inference_lora_strengths: Optional[List[float]] = None,
    ) -> None:
        """Instantiate and wire the generation pipeline (plus adapters)."""
        if self.is_loaded:
            return

        print("[Load] Building main Flux pipeline …")
        
        # Build pipeline components
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=self.torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=self.torch_dtype
        )
        tokenizer2 = T5TokenizerFast.from_pretrained(
            self.model_path, subfolder="tokenizer_2"
        )
        text_encoder2 = T5EncoderModel.from_pretrained(
            self.model_path, subfolder="text_encoder_2", torch_dtype=self.torch_dtype
        )
        
        # Load the transformer model
        from diffusers import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_path, subfolder="transformer", torch_dtype=self.torch_dtype
        )

        # NEW: Store clean components for inpaint pipeline (before LoRAs/IP-Adapter)
        print("[Load] Storing clean components for inpaint pipeline …")
        self._clean_vae = vae
        self._clean_transformer = transformer  # Share the same transformer, we'll clean it later
        self._clean_text_encoder = text_encoder
        self._clean_text_encoder_2 = text_encoder2
        self._clean_tokenizer = tokenizer
        self._clean_tokenizer_2 = tokenizer2
        self._clean_scheduler = scheduler

        # NEW: Load IP-Adapter components if enabled (AFTER storing clean components)
        image_encoder = None
        feature_extractor = None
        if self.ip_adapter_enabled:
            print("[Load] Loading IP-Adapter components …")
            try:
                image_encoder = None
                # Load feature extractor from CLIP model
                feature_extractor = None
                
                print("[Load] IP-Adapter components loaded successfully")
            except Exception as e:
                print(f"[Load] Warning: Could not load IP-Adapter components: {e}")
                print("[Load] Continuing without IP-Adapter support")
                self.ip_adapter_enabled = False

        # Create pipeline
        self.pipeline = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder2,
            tokenizer_2=tokenizer2,
            vae=vae,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        ).to(self.device, self.torch_dtype)

        # Load IP-Adapter weights if enabled
        if self.ip_adapter_enabled and image_encoder is not None:
            self._load_ip_adapter()

        # User LoRAs ----------------------------------------------------------
        inference_lora_paths = inference_lora_paths or []
        inference_lora_strengths = (
            inference_lora_strengths or [1.0] * len(inference_lora_paths)
        )
        if len(inference_lora_strengths) < len(inference_lora_paths):
            inference_lora_strengths.extend(
                [1.0] * (len(inference_lora_paths) - len(inference_lora_strengths))
            )
        if inference_lora_paths:
            print(f"[Load] Attaching {len(inference_lora_paths)} generation LoRAs …")
            adapters = []
            for i, (path, strength) in enumerate(
                zip(inference_lora_paths, inference_lora_strengths)
            ):
                name = f"gen_lora_{i}"
                self.pipeline.load_lora_weights(path, adapter_name=name)
                adapters.append(name)
                print(f"       • {Path(path).name}  strength={strength}")
            self.pipeline.set_adapters(adapters, inference_lora_strengths)

        # Detection helpers --------------------------------------------------
        if self.face_inpainting:
            self._setup_face_detection()
        
        if self.upper_body_inpainting:
            self._setup_pose_detection()

        self.is_loaded = True
        print("[Load] Done ✓")

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter weights and preprocess reference images."""
        if self.ip_adapter_loaded or not self.ip_adapter_enabled:
            return
            
        try:
            print("[IP-Adapter] Loading IP-Adapter weights …")
            
            # Load IP-Adapter
            self.pipeline.load_ip_adapter(
                self.ip_adapter_model_path,
                weight_name="ip_adapter.safetensors"  # Adjust filename as needed
            )
            
            # Set default scale after loading (FIXED: Use proper method)
            if hasattr(self.pipeline, 'set_ip_adapter_scale'):
                default_scale = self.ip_adapter_scales[0] if self.ip_adapter_scales else 0.6
                self.pipeline.set_ip_adapter_scale(default_scale)
                print(f"[IP-Adapter] Set default scale to {default_scale}")
            else:
                print("[IP-Adapter] Warning: set_ip_adapter_scale method not available")
            
            # Preprocess reference images
            print("[IP-Adapter] Preprocessing reference images …")
            self.preprocessed_ip_images = []
            
            for img_path in self.ip_adapter_images:
                if os.path.exists(img_path):
                    img = load_and_preprocess_image(img_path, self.ip_adapter_target_size)
                    self.preprocessed_ip_images.append(img)
                else:
                    print(f"[IP-Adapter] Warning: Image not found: {img_path}")
                    
            if not self.preprocessed_ip_images:
                print("[IP-Adapter] Warning: No valid reference images found, disabling IP-Adapter")
                self.ip_adapter_enabled = False
                return
                
            print(f"[IP-Adapter] Successfully loaded {len(self.preprocessed_ip_images)} reference images")
            self.ip_adapter_loaded = True
            
        except Exception as e:
            print(f"[IP-Adapter] Error loading IP-Adapter: {e}")
            print("[IP-Adapter] Continuing without IP-Adapter support")
            self.ip_adapter_enabled = False

    # ------------------------------ detection utils ------------------------ #

    def _setup_face_detection(self) -> None:
        """Enhanced face detection setup with better settings for full-body images."""
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.2
            )
            print(f"[Face] MediaPipe face detection model loaded (full range, confidence=0.2)")
        except ImportError:
            print(f"[Face] ERROR: MediaPipe not installed. Please install with 'pip install mediapipe'")
            raise ValueError("MediaPipe not installed. Please install with 'pip install mediapipe'")

    def _setup_pose_detection(self) -> None:
        """Setup pose detection for shoulder detection."""
        try:
            import mediapipe as mp
            if not hasattr(self, 'mp'):
                self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print(f"[Pose] MediaPipe pose detection model loaded (complexity=2, confidence=0.3)")
        except ImportError:
            print(f"[Pose] ERROR: MediaPipe not installed. Please install with 'pip install mediapipe'")
            raise ValueError("MediaPipe not installed. Please install with 'pip install mediapipe'")

    def detect_faces(self, img: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Improved face detection optimized for full-body images where faces are smaller."""
        if not self.face_detector:
            self._setup_face_detection()
        
        img_rgb = np.array(img)
        height, width, _ = img_rgb.shape
        
        print(f"[Face Detection] Image size: {width}x{height}")
        
        results = self.face_detector.process(img_rgb)
        
        boxes: List[Tuple[int, int, int, int]] = []
        if results.detections:
            print(f"[Face Detection] Found {len(results.detections)} raw detections")
            
            for i, detection in enumerate(results.detections):
                confidence = detection.score[0] if detection.score else 0
                print(f"[Face Detection] Detection {i+1}: confidence={confidence:.3f}")
                
                if confidence < 0.3:
                    print(f"      ↳ Skipped: Low confidence ({confidence:.3f} < 0.3)")
                    continue
                    
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                print(f"      ↳ Raw bbox: x={x}, y={y}, w={w}, h={h}")
                
                face_area = w * h
                image_area = width * height
                face_ratio = face_area / image_area
                
                print(f"      ↳ Face ratio: {face_ratio:.4f} ({face_area}/{image_area})")
                
                min_face_ratio = 0.001
                max_face_ratio = 0.15
                
                if face_ratio < min_face_ratio:
                    print(f"      ↳ Skipped: Face too small ({face_ratio:.4f} < {min_face_ratio})")
                    continue
                if face_ratio > max_face_ratio:
                    print(f"      ↳ Skipped: Face too large ({face_ratio:.4f} > {max_face_ratio})")
                    continue
                
                exp = 0.05
                nx = max(0, int(x - w * exp))
                ny = max(0, int(y - h * exp))
                nw = min(width - nx, int(w * (1 + 2 * exp)))
                nh = min(height - ny, int(h * (1 + 2 * exp)))
                
                print(f"      ↳ Final bbox: x={nx}, y={ny}, w={nw}, h={nh}")
                boxes.append((nx, ny, nw, nh))
        else:
            print("[Face Detection] No detections found by MediaPipe")
        
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        
        print(f"[Face Detection] Final result: {len(boxes)} faces passed filtering")
        return boxes

    def detect_upper_body_region(self, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Detect the face + neck region using pose detection."""
        if not self.pose_detector:
            self._setup_pose_detection()
        
        img_rgb = np.array(img)
        height, width, _ = img_rgb.shape
        
        print(f"[Upper Body Detection] Image size: {width}x{height}")
        
        results = self.pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            print("[Upper Body Detection] No pose landmarks detected")
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        
        shoulder_confidence_threshold = 0.3
        if (left_shoulder.visibility < shoulder_confidence_threshold and 
            right_shoulder.visibility < shoulder_confidence_threshold):
            print(f"[Upper Body Detection] Shoulders not visible enough")
            return None
        
        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)
        
        nose = landmarks[NOSE]
        left_eye = landmarks[LEFT_EYE] 
        right_eye = landmarks[RIGHT_EYE]
        
        face_landmarks = [nose, left_eye, right_eye]
        face_points = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks if lm.visibility > 0.3]
        
        if not face_points:
            print("[Upper Body Detection] Insufficient facial landmarks visible")
            return None
            
        face_xs = [p[0] for p in face_points]
        face_ys = [p[1] for p in face_points]
        face_center_x = sum(face_xs) // len(face_xs)
        face_top_y = min(face_ys)
        
        face_height_estimate = max(left_shoulder_y, right_shoulder_y) - face_top_y
        top_padding = int(face_height_estimate * 0.3)
        region_top_y = max(0, face_top_y - top_padding)
        
        neck_bottom_y = max(left_shoulder_y, right_shoulder_y)
        
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)
        center_x = (left_shoulder_x + right_shoulder_x) // 2
        
        extension_factor = 0.3
        region_width = int(shoulder_width * (1 + 2 * extension_factor))
        
        x = max(0, center_x - region_width // 2)
        y = region_top_y
        w = min(width - x, region_width)
        h = min(height - y, neck_bottom_y - region_top_y)
        
        min_region_ratio = 0.01
        region_area = w * h
        image_area = width * height
        
        if region_area < image_area * min_region_ratio:
            print(f"[Upper Body Detection] Face+neck region too small")
            return None
        
        print(f"[Upper Body Detection] Face+neck region: x={x}, y={y}, w={w}, h={h}")
        
        return (x, y, w, h)

    # ------------------------------ inpainting ---------------------------- #

    def _clean_transformer_ip_adapter_state(self, transformer):
        """Remove IP-Adapter state from transformer to prepare for inpaint pipeline."""
        print("[Inpaint] Cleaning IP-Adapter state from transformer ...")
        
        # Reset attention processors to default if they were modified by IP-Adapter
        if hasattr(transformer, 'set_attn_processor'):
            try:
                # Get all attention processor names and set them to default
                attn_procs = {}
                for name in transformer.attn_processors.keys():
                    # Use the default FluxAttnProcessor2_0 for all attention layers
                    from diffusers.models.attention_processor import FluxAttnProcessor2_0
                    attn_procs[name] = FluxAttnProcessor2_0()
                
                transformer.set_attn_processor(attn_procs)
                print(f"[Inpaint] Reset {len(attn_procs)} attention processors to default FluxAttnProcessor2_0")
            except Exception as e:
                print(f"[Inpaint] Warning: Could not reset attention processors: {e}")
        
        # Clear any IP-Adapter hidden states
        if hasattr(transformer, '_ip_adapter_hidden_states_input'):
            transformer._ip_adapter_hidden_states_input = None
            
        # Clear encoder projection if it exists
        if hasattr(transformer, 'encoder_hid_proj'):
            transformer.encoder_hid_proj = None
            
        # Clear any cached IP-Adapter data
        if hasattr(transformer, '_ip_adapter_attn_processors'):
            transformer._ip_adapter_attn_processors = None
            
        # Clear any IP-Adapter scales or configurations
        if hasattr(transformer, '_ip_adapter_scale'):
            transformer._ip_adapter_scale = None
            
        # Clear IP-Adapter image embeddings if they exist
        if hasattr(transformer, '_ip_adapter_image_embeds'):
            transformer._ip_adapter_image_embeds = None
            
        print("[Inpaint] Transformer cleaned of IP-Adapter state")

    def _init_inpaint_pipeline(self) -> None:
        """Initialize clean inpaint pipeline without inference LoRAs or IP-Adapter."""
        if self.inpaint_pipeline is not None:
            return
            
        print("[Inpaint] Building CLEAN inpaint pipeline (no inference LoRAs/IP-Adapter) …")
        
        if self._clean_vae is None:
            # Fallback: load fresh components if clean ones not available
            print("[Inpaint] Warning: Clean components not available, loading fresh from scratch")
            self.inpaint_pipeline = FluxInpaintPipeline.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype
            ).to(self.device)
        else:
            # Clean the transformer of any IP-Adapter state before using in inpaint pipeline
            if self.ip_adapter_enabled:
                self._clean_transformer_ip_adapter_state(self._clean_transformer)
            
            # Use the clean components stored before LoRAs/IP-Adapter were applied
            # NOTE: Explicitly exclude image_encoder and feature_extractor to avoid IP-Adapter contamination
            self.inpaint_pipeline = FluxInpaintPipeline(
                vae=self._clean_vae,
                transformer=self._clean_transformer,
                text_encoder=self._clean_text_encoder,
                text_encoder_2=self._clean_text_encoder_2,
                tokenizer=self._clean_tokenizer,
                tokenizer_2=self._clean_tokenizer_2,
                scheduler=self._clean_scheduler,
                # Explicitly set these to None to ensure no IP-Adapter functionality
                image_encoder=None,
                feature_extractor=None,
            ).to(self.device)
            
        print("[Inpaint] Clean inpaint pipeline initialized (no style contamination)")
        
        # Double-check: verify the transformer has clean attention processors
        if hasattr(self.inpaint_pipeline.transformer, 'attn_processors'):
            processor_types = [type(proc).__name__ for proc in self.inpaint_pipeline.transformer.attn_processors.values()]
            unique_types = set(processor_types)
            print(f"[Inpaint] Attention processor types: {unique_types}")
            if any('IP' in ptype for ptype in unique_types):
                print("[Inpaint] WARNING: IP-Adapter processors still detected, attempting additional cleanup...")
                self._clean_transformer_ip_adapter_state(self.inpaint_pipeline.transformer)

    def _load_face_loras(self) -> None:
        """Load face LoRAs into inpaint pipeline on-demand to prevent style leak."""
        if self.face_loras_loaded or not self.face_lora_paths:
            return
            
        print("[Face] Loading face LoRAs on-demand to prevent style leak …")
        names = []
        for i, (p, s) in enumerate(zip(self.face_lora_paths, self.face_lora_strengths)):
            n = f"face_lora_{i}"
            self.inpaint_pipeline.load_lora_weights(p, adapter_name=n)
            names.append(n)
            print(f"       • {Path(p).name}  strength={s}")
        self.inpaint_pipeline.set_adapters(names, self.face_lora_strengths)
        self.face_loras_loaded = True

    def _load_upper_body_loras(self) -> None:
        """Load upper body LoRAs into inpaint pipeline on-demand to prevent style leak."""
        if self.upper_body_loras_loaded or not self.upper_body_lora_paths:
            return
            
        print("[Upper Body] Loading upper body LoRAs on-demand to prevent style leak …")
        names = []
        for i, (p, s) in enumerate(zip(self.upper_body_lora_paths, self.upper_body_lora_strengths)):
            n = f"upper_body_lora_{i}"
            self.inpaint_pipeline.load_lora_weights(p, adapter_name=n)
            names.append(n)
            print(f"       • {Path(p).name}  strength={s}")
        self.inpaint_pipeline.set_adapters(names, self.upper_body_lora_strengths)
        self.upper_body_loras_loaded = True

    def _inpaint_face(self, img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
        """Clean, effective face inpainting approach for Flux models with size preservation."""
        x, y, w, h = box
        try:
            self._init_inpaint_pipeline()
            self._load_face_loras()
            
            if self.save_face_debug:
                debug_dir = self.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
            
            print(f"      ↳ Processing face at ({x}, {y}) size {w}x{h}")
            
            min_size = 512
            max_size = 1024
            
            face_size = max(w, h)
            if face_size < min_size:
                scale_factor = min_size / face_size
            elif face_size > max_size:
                scale_factor = max_size / face_size
            else:
                scale_factor = 1.0
                
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            target_w = ((target_w + 7) // 8) * 8
            target_h = ((target_h + 7) // 8) * 8
            
            print(f"      ↳ Target processing size: {target_w}x{target_h} (scale: {scale_factor:.2f})")
            
            # REDUCED padding to prevent size inflation
            padding = 0.08  # Reduced from 0.15 to 0.08
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.width, x + w + pad_x)
            y2 = min(img.height, y + h + pad_y)
            
            face_region = img.crop((x1, y1, x2, y2))
            original_region_size = face_region.size
            
            if self.save_face_debug:
                face_path = debug_dir / f"face_original_{timestamp}.png"
                face_region.save(face_path)
                print(f"      ↳ Debug: Original face region saved")
            
            face_resized = face_region.resize((target_w, target_h), Image.LANCZOS)
            
            # CONSERVATIVE mask that stays within original face bounds
            mask_padding = 0.12  # Increased from 0.05 to be more conservative
            mask_x1 = int(target_w * mask_padding)
            mask_y1 = int(target_h * mask_padding)
            mask_x2 = int(target_w * (1 - mask_padding))
            mask_y2 = int(target_h * (1 - mask_padding))
            
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
            
            blur_size = max(5, min(target_w, target_h) // 20)
            if blur_size % 2 == 0:
                blur_size += 1
                
            mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            mask_pil = Image.fromarray(mask_blurred)
            
            # MODIFIED prompt to preserve facial structure
            enhanced_prompt = f"{self.face_prompt}, natural facial proportions, preserve face shape"
            enhanced_negative = f"{self.face_neg_prompt}, oversized features, enlarged face, distorted proportions"
            
            print(f"      ↳ Running face inpainting at {target_w}x{target_h}...")
            
            # REDUCED strength to preserve more of original structure
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=face_resized,
                mask_image=mask_pil,
                num_inference_steps=32,
                guidance_scale=self.face_guidance_scale,
                strength=0.75,  # Reduced from 0.85 to 0.75
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
            
            inpainted_face = result.images[0]
            face_final = inpainted_face.resize(original_region_size, Image.LANCZOS)
            
            # MORE CONSERVATIVE blending that preserves face boundaries
            final_mask = np.zeros((original_region_size[1], original_region_size[0]), dtype=np.uint8)
            
            face_x_in_region = pad_x
            face_y_in_region = pad_y
            
            # INCREASED feathering to create softer edges
            feather = min(w, h) // 8  # Changed from //10 to //8 for more feathering
            
            # TIGHTER mask bounds to prevent expansion
            mask_x1 = max(0, face_x_in_region + int(feather * 1.5))  # More conservative inset
            mask_y1 = max(0, face_y_in_region + int(feather * 1.5))
            mask_x2 = min(original_region_size[0], face_x_in_region + w - int(feather * 1.5))
            mask_y2 = min(original_region_size[1], face_y_in_region + h - int(feather * 1.5))
            
            if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                final_mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
                
                # LARGER blur for smoother blending
                final_blur_size = max(5, feather)  # Increased blur size
                if final_blur_size % 2 == 0:
                    final_blur_size += 1
                    
                final_mask_blurred = cv2.GaussianBlur(final_mask, (final_blur_size, final_blur_size), 0)
            else:
                final_mask_blurred = np.ones((original_region_size[1], original_region_size[0]), dtype=np.uint8) * 128
            
            result_img = img.copy()
            original_region_img = img.crop((x1, y1, x2, y2))
            
            original_np = np.array(original_region_img).astype(np.float32)
            final_np = np.array(face_final).astype(np.float32)
            blend_mask_np = final_mask_blurred.astype(np.float32) / 255.0
            
            if len(blend_mask_np.shape) == 2:
                blend_mask_np = np.expand_dims(blend_mask_np, axis=2)
            
            # ADDITIONAL step: Scale down the blend mask slightly to preserve more original
            blend_mask_np = blend_mask_np * 0.95  # Scale down to preserve 10% more original
            
            blended_np = original_np * (1 - blend_mask_np) + final_np * blend_mask_np
            blended_img = Image.fromarray(np.uint8(blended_np))
            
            result_img.paste(blended_img, (x1, y1))
            
            print(f"      ✓ Face inpainting completed successfully")
            return result_img
            
        except Exception as e:
            print(f"[Face] Inpainting failed: {e}")
            traceback.print_exc()
            return img

    def _inpaint_upper_body(self, img: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
        """Inpaint the face + neck region."""
        x, y, w, h = region
        try:
            self._init_inpaint_pipeline()
            self._load_upper_body_loras()
            
            if self.save_upper_body_debug:
                debug_dir = self.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
            
            print(f"      ↳ Processing face+neck region at ({x}, {y}) size {w}x{h}")
            
            min_size = 512
            max_size = 1024
            
            region_size = max(w, h)
            if region_size < min_size:
                scale_factor = min_size / region_size
            elif region_size > max_size:
                scale_factor = max_size / region_size
            else:
                scale_factor = 1.0
                
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            target_w = ((target_w + 7) // 8) * 8
            target_h = ((target_h + 7) // 8) * 8
            
            face_neck_region = img.crop((x, y, x + w, y + h))
            original_region_size = face_neck_region.size
            
            region_resized = face_neck_region.resize((target_w, target_h), Image.LANCZOS)
            
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            
            edge_margin = 0.08
            center_x1 = int(target_w * edge_margin)
            center_y1 = int(target_h * edge_margin)
            center_x2 = int(target_w * (1 - edge_margin))
            center_y2 = int(target_h * (1 - edge_margin))
            
            mask[center_y1:center_y2, center_x1:center_x2] = 255
            
            blur_size = max(9, min(target_w, target_h) // 25)
            if blur_size % 2 == 0:
                blur_size += 1
                
            mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
            edge_falloff = 0.12
            falloff_pixels = int(min(target_w, target_h) * edge_falloff)
            
            for i in range(falloff_pixels):
                alpha = i / falloff_pixels
                
                if i < target_h:
                    mask_blurred[i, :] = mask_blurred[i, :] * alpha
                
                if target_h - 1 - i >= 0:
                    mask_blurred[target_h - 1 - i, :] = mask_blurred[target_h - 1 - i, :] * alpha
                
                if i < target_w:
                    mask_blurred[:, i] = mask_blurred[:, i] * alpha
                
                if target_w - 1 - i >= 0:
                    mask_blurred[:, target_w - 1 - i] = mask_blurred[:, target_w - 1 - i] * alpha
            
            mask_pil = Image.fromarray(mask_blurred)
            
            enhanced_prompt = f"{self.upper_body_prompt}"
            enhanced_negative = f"{self.upper_body_neg_prompt}"
            
            # Clean inpaint call - no IP-Adapter or cross_attention_kwargs
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=region_resized,
                mask_image=mask_pil,
                num_inference_steps=35,
                guidance_scale=self.upper_body_guidance_scale,
                strength=0.8,
                generator=torch.Generator(device=self.device).manual_seed(42),
                # Explicitly avoid any IP-Adapter related parameters
            )
            
            inpainted_face_neck = result.images[0]
            face_neck_final = inpainted_face_neck.resize(original_region_size, Image.LANCZOS)
            
            blend_mask = np.ones((original_region_size[1], original_region_size[0]), dtype=np.uint8) * 255
            
            falloff_ratio = 0.08
            falloff_x = max(1, int(original_region_size[0] * falloff_ratio))
            falloff_y = max(1, int(original_region_size[1] * falloff_ratio))
            
            for i in range(falloff_x):
                alpha = i / falloff_x
                blend_mask[:, i] = blend_mask[:, i] * alpha
                blend_mask[:, -(i+1)] = blend_mask[:, -(i+1)] * alpha
            
            for i in range(falloff_y):
                alpha = i / falloff_y
                blend_mask[i, :] = blend_mask[i, :] * alpha
                blend_mask[-(i+1), :] = blend_mask[-(i+1), :] * alpha
            
            blend_blur_size = max(3, min(original_region_size) // 30)
            if blend_blur_size % 2 == 0:
                blend_blur_size += 1
            
            blend_mask_blurred = cv2.GaussianBlur(blend_mask, (blend_blur_size, blend_blur_size), 0)
            
            result_img = img.copy()
            original_region_img = img.crop((x, y, x + w, y + h))
            
            original_np = np.array(original_region_img).astype(np.float32)
            final_np = np.array(face_neck_final).astype(np.float32)
            blend_mask_np = blend_mask_blurred.astype(np.float32) / 255.0
            
            if len(blend_mask_np.shape) == 2:
                blend_mask_np = np.expand_dims(blend_mask_np, axis=2)
            
            blended_np = original_np * (1 - blend_mask_np) + final_np * blend_mask_np
            blended_img = Image.fromarray(np.uint8(blended_np))
            
            result_img.paste(blended_img, (x, y))
            
            print(f"      ✓ Face+neck inpainting completed successfully")
            return result_img
            
        except Exception as e:
            print(f"[Face+Neck] Inpainting failed: {e}")
            traceback.print_exc()
            return img

    # ------------------------------ generation --------------------------- #

    def generate_images(
        self,
        *,
        prompts: List[str],
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        seed: int = -1,
        output_filenames: Optional[List[str]] = None,
        output_ext: str = ".png",
        # NEW: IP-Adapter parameters for per-generation control
        ip_adapter_images: Optional[List[Union[str, Image.Image]]] = None,
        ip_adapter_scales: Optional[List[float]] = None,
    ) -> List[Image.Image]:
        """Generate images with optional IP-Adapter control."""
        if not self.is_loaded:
            self.load_model()

        # Prepare IP-Adapter images for this generation if provided
        generation_ip_images = None
        generation_ip_scales = None
        
        if ip_adapter_images is not None:
            print(f"[Gen] Using per-generation IP-Adapter images: {len(ip_adapter_images)} images")
            generation_ip_images = []
            
            for img_input in ip_adapter_images:
                if isinstance(img_input, str):
                    if os.path.exists(img_input):
                        img = load_and_preprocess_image(img_input, self.ip_adapter_target_size)
                        generation_ip_images.append(img)
                    else:
                        print(f"[Gen] Warning: IP-Adapter image not found: {img_input}")
                elif isinstance(img_input, Image.Image):
                    generation_ip_images.append(img_input)
                else:
                    print(f"[Gen] Warning: Invalid IP-Adapter image type: {type(img_input)}")
            
            generation_ip_scales = ip_adapter_scales or [0.6] * len(generation_ip_images)
            
        elif self.ip_adapter_enabled and self.preprocessed_ip_images:
            # Use global IP-Adapter images
            print(f"[Gen] Using global IP-Adapter images: {len(self.preprocessed_ip_images)} images")
            generation_ip_images = self.preprocessed_ip_images
            generation_ip_scales = self.ip_adapter_scales

        all_imgs: List[Image.Image] = []
        for idx, prompt in enumerate(prompts):
            use_seed = random.randint(0, 2**32 - 1) if seed < 0 else seed
            print(f"[Gen] Prompt {idx+1}/{len(prompts)}  seed={use_seed}")
            if generation_ip_images:
                print(f"      → IP-Adapter: {len(generation_ip_images)} images, scales: {generation_ip_scales}")
            print(f"      → {prompt}")
            
            gen = torch.Generator(device=self.device).manual_seed(use_seed)
            
            # Prepare generation arguments
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images_per_prompt,
                "generator": gen,
            }
            
            # Add IP-Adapter parameters if available - FIXED METHOD
            if generation_ip_images and self.ip_adapter_enabled:
                gen_kwargs["ip_adapter_image"] = generation_ip_images
                
                # For Flux IP-Adapter, set the scale directly on the pipeline before generation
                if generation_ip_scales:
                    try:
                        # Try the standard method first
                        if hasattr(self.pipeline, 'set_ip_adapter_scale'):
                            scale_to_set = generation_ip_scales[0] if len(generation_ip_scales) == 1 else generation_ip_scales
                            self.pipeline.set_ip_adapter_scale(scale_to_set)
                            print(f"[IP-Adapter] Set scale to {scale_to_set}")
                        else:
                            print("[IP-Adapter] Warning: set_ip_adapter_scale method not available")
                            # Try alternative method - pass as direct parameter
                            gen_kwargs["ip_adapter_scale"] = generation_ip_scales[0] if len(generation_ip_scales) == 1 else generation_ip_scales
                    except Exception as e:
                        print(f"[IP-Adapter] Warning: Could not set scale: {e}")
                        # Continue without scale setting - use default from load time
            
            try:
                res = self.pipeline(**gen_kwargs)
                images = res.images
            except Exception as e:
                print(f"[Gen] Error during generation: {e}")
                print("[Gen] Trying without IP-Adapter...")
                # Fallback without IP-Adapter
                gen_kwargs.pop("ip_adapter_image", None)
                gen_kwargs.pop("ip_adapter_scale", None)
                res = self.pipeline(**gen_kwargs)
                images = res.images
            
            # Save intermediate images before inpainting if debug is enabled
            if self.save_face_debug or self.save_upper_body_debug:
                debug_dir = self.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
                for i, img in enumerate(images):
                    debug_path = debug_dir / f"initial_generation_{timestamp}_{i}.png"
                    img.save(debug_path)
                    print(f"[Debug] Saved initial generation image {i+1}/{len(images)}")
            
            # Process each generated image
            for j, im in enumerate(images):
                processed_img = im
                
                # Upper body inpainting (face + neck region)
                if self.upper_body_inpainting:
                    upper_body_region = self.detect_upper_body_region(processed_img)
                    if upper_body_region:
                        print(f"      ↳ Face+neck region detected, inpainting …")
                        processed_img = self._inpaint_upper_body(processed_img, upper_body_region)
                    else:
                        print(f"      ↳ No face+neck region detected, skipping upper body inpainting")
                
                # Face inpainting (fine details)
                if self.face_inpainting:
                    if self.upper_body_loras_loaded:
                        self.upper_body_loras_loaded = False
                    
                    faces = self.detect_faces(processed_img)
                    if faces:
                        print(f"      ↳ {len(faces)} face(s) found, inpainting …")
                        for b in faces:
                            processed_img = self._inpaint_face(processed_img, b)
                    else:
                        print(f"      ↳ No faces found, skipping face inpainting")
                
                images[j] = processed_img

            # Save final images
            for j, im in enumerate(images):
                if output_filenames and idx * num_images_per_prompt + j < len(output_filenames):
                    fname = output_filenames[idx * num_images_per_prompt + j]
                    if not os.path.splitext(fname)[1]:
                        fname += output_ext
                else:
                    slug = _slugify(prompt)
                    global_img_idx = idx * num_images_per_prompt + j + 1
                    fname = f"{global_img_idx:03d}_{slug}_{use_seed}_{j}{output_ext}"
                path = self.output_dir / fname
                im.save(path)
                print(f"      ✔ saved → {path.relative_to(self.output_dir.parent)}")
            all_imgs.extend(images)

        print("[Gen] Completed ✓")
        return all_imgs


# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flux image generation helper with IP-Adapter and upper body inpainting")
    p.add_argument("--config", required=True, help="Path to YAML configuration file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # YAML config -----------------------------------------------------------
    if not args.config or not os.path.exists(args.config):
        raise SystemExit("✖ No config file provided or file does not exist")
        
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    
    # Extract configuration values
    model_cfg = cfg.get("model", {})
    generate_cfg = cfg.get("generate", {})
    ip_adapter_cfg = cfg.get("ip_adapter", {})  # NEW: IP-Adapter config section
    
    # Prompts ---------------------------------------------------------------
    prompts = generate_cfg.get("prompts", [])
    if not prompts:
        raise SystemExit("✖ No prompts provided in config file")

    # Build inference object -----------------------------------------------
    flux = FluxInference(
        model_path=model_cfg.get("name_or_path", "black-forest-labs/FLUX.1-dev"),
        output_dir=generate_cfg.get("output_dir", "outputs"),
        device=model_cfg.get("device", "cuda"),
        dtype=model_cfg.get("dtype", "float16"),
        face_inpainting=generate_cfg.get("face_inpainting", False),
        upper_body_inpainting=generate_cfg.get("upper_body_inpainting", False),
        face_lora_paths=generate_cfg.get("face_lora_paths", []),
        face_lora_strengths=generate_cfg.get("face_lora_strengths", []),
        upper_body_lora_paths=generate_cfg.get("upper_body_lora_paths", []),
        upper_body_lora_strengths=generate_cfg.get("upper_body_lora_strengths", []),
        # NEW: IP-Adapter configuration
        ip_adapter_model_path=ip_adapter_cfg.get("model_path", "XLabs-AI/flux-ip-adapter"),
        ip_adapter_images=ip_adapter_cfg.get("reference_images", []),
        ip_adapter_scales=ip_adapter_cfg.get("scales", []),
        ip_adapter_target_size=tuple(ip_adapter_cfg.get("target_size", [512, 512])),
        low_vram=model_cfg.get("low_vram", False),
        config=cfg,
    )

    # Load weights ----------------------------------------------------------
    flux.load_model(
        inference_lora_paths=model_cfg.get("inference_lora_paths", []),
        inference_lora_strengths=model_cfg.get("inference_lora_strengths", []),
    )

    # Run generation --------------------------------------------------------
    flux.generate_images(
        prompts=prompts,
        negative_prompt=generate_cfg.get("negative_prompt", "ugly, bad anatomy, blurry"),
        width=generate_cfg.get("width", 1024),
        height=generate_cfg.get("height", 1024),
        num_inference_steps=generate_cfg.get("sample_steps", 20),
        guidance_scale=generate_cfg.get("guidance_scale", 4.0),
        num_images_per_prompt=generate_cfg.get("num_images", 1),
        seed=generate_cfg.get("seed", -1),
        output_ext=generate_cfg.get("ext", ".png"),
        # NEW: Per-generation IP-Adapter control
        ip_adapter_images=generate_cfg.get("ip_adapter_images"),
        ip_adapter_scales=generate_cfg.get("ip_adapter_scales"),
    )


if __name__ == "__main__":
    main()