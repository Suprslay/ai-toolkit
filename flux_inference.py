#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory-Optimized Flux Inference with Adapter Management + Direct Inpainting
---------------------------------------------------------------------------
A single‑file utility for running image generation with Flux models.
Features
~~~~~~~~
* Memory-efficient adapter management (shared transformer + on-demand adapter loading)
* IP-Adapter support for garment/style control
* Character LoRA + Garment IP-Adapter combination
* Optional face detection + in‑place inpainting with adapter switching
* Upper body inpainting (face + neck only)
* DIRECT INPAINTING: Process existing images without generation
* YAML/CLI hybrid configuration (CLI overrides YAML)
* VRAM‑friendly switches
* Robust filename handling and prompt logging

Usage
~~~~~
python flux_inference.py --config config.yaml

NEW: Direct Inpainting Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set skip_generation: true and start_image: "path/to/image.jpg" in config to process
existing images directly with face/upper body inpainting.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import copy

import cv2
import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from diffusers import (
    AutoencoderKL,
    FluxInpaintPipeline,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
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

def get_torch_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_str = dtype_str.lower()
    if dtype_str in ["float16", "fp16", "half"]:
        return torch.float16
    elif dtype_str in ["bfloat16", "bf16"]:
        return torch.bfloat16
    elif dtype_str in ["float32", "fp32", "full"]:
        return torch.float32
    else:
        print(f"[Warning] Unknown dtype '{dtype_str}', defaulting to float16")
        return torch.float16


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
    """Memory-efficient manager around Diffusers' `FluxPipeline` with adapter switching."""

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
        # IP-Adapter configuration
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
        self.torch_dtype = get_torch_dtype(dtype)
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
        self.face_prompt = "slaymnsha A woman, smooth skin, make-up, confident, lightly smiling, model, fashion"
        self.face_neg_prompt = (
            "deformed face, distorted face, disfigured, low res, bad anatomy, cartoon, anime, painting, sketch, low quality, pixelated, jpeg artifacts, oversaturated"
        )
        self.face_guidance_scale = 4.0

        # Upper body‑related opts
        self.upper_body_inpainting = upper_body_inpainting
        self.upper_body_lora_paths = upper_body_lora_paths or []
        self.upper_body_lora_strengths = upper_body_lora_strengths or [0.75] * len(self.upper_body_lora_paths)
        if len(self.upper_body_lora_strengths) < len(self.upper_body_lora_paths):
            self.upper_body_lora_strengths.extend(
                [0.75] * (len(self.upper_body_lora_paths) - len(self.upper_body_lora_strengths))
            )
        self.upper_body_prompt = "slaymnsha a woman, beautiful face, elegant neck, portrait, smooth skin, professional photography, high quality"
        self.upper_body_neg_prompt = (
            "deformed face, distorted face, disfigured, low res, bad anatomy, cartoon, anime, painting, sketch, low quality, pixelated, jpeg artifacts, oversaturated"
        )
        self.upper_body_guidance_scale = 3.5

        # IP-Adapter configuration
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

        # MEMORY OPTIMIZED: Single pipeline with adapter switching
        self.generation_pipeline: FluxPipeline | None = None
        self.inpaint_pipeline: FluxInpaintPipeline | None = None
        
        # Detection utilities
        self.face_detector = None
        self.pose_detector = None
        
        # State tracking
        self.is_loaded = False
        self.preprocessed_ip_images = []
        
        # MEMORY OPTIMIZED: Track adapters for switching (not separate pipelines)
        self.generation_adapters = []
        self.face_adapters = []
        self.upper_body_adapters = []
        self.current_adapters = None  # Track which adapters are currently active
        
        # Add missing imports and state tracking
        self.face_loras_loaded = False
        self.upper_body_loras_loaded = False

        print(f"[Init] model_path={model_path}  device={device}  dtype={dtype}")
        print(f"[Init] MEMORY OPTIMIZED: Using adapter switching instead of separate pipelines")
        
        if self.ip_adapter_enabled:
            print(f"[Init] IP-Adapter ENABLED with {len(self.ip_adapter_images)} reference images")
            for img_path, scale in zip(self.ip_adapter_images, self.ip_adapter_scales):
                print(f"       • {Path(img_path).name}  scale={scale}")
        
        if face_inpainting:
            print("[Init] Face inpainting ENABLED")
            if self.face_lora_paths:
                for p, s in zip(self.face_lora_paths, self.face_lora_strengths):
                    print(f"       • {Path(p).name}  strength={s} (on-demand loading)")
        
        if upper_body_inpainting:
            print("[Init] Upper body inpainting ENABLED")
            print("       → Upper body = FACE + NECK region only")
            if self.upper_body_lora_paths:
                for p, s in zip(self.upper_body_lora_paths, self.upper_body_lora_strengths):
                    print(f"       • {Path(p).name}  strength={s} (on-demand loading)")

    # ------------------------------ model loading -------------------------- #

    def _load_base_components(self):
        """Load the base model components - MEMORY OPTIMIZED to load once."""
        print("[Load] Loading base Flux components (ONCE for memory efficiency) …")
        
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
        
        # MEMORY OPTIMIZED: Load transformer only once
        print("[Load] Loading transformer (SINGLE INSTANCE) …")
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_path, subfolder="transformer", torch_dtype=self.torch_dtype
        )
        
        return {
            'scheduler': scheduler,
            'vae': vae,
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'tokenizer_2': tokenizer2,
            'text_encoder_2': text_encoder2,
            'transformer': transformer,  # Single instance shared
        }

    def load_model(
        self,
        *,
        inference_lora_paths: Optional[List[str]] = None,
        inference_lora_strengths: Optional[List[float]] = None,
        skip_generation: bool = False,
    ) -> None:
        """Load pipelines with memory-optimized adapter management."""
        if self.is_loaded:
            return

        print("[Load] Building MEMORY OPTIMIZED Flux pipelines …")
        
        # Load base components once
        base_components = self._load_base_components()
        
        # MEMORY OPTIMIZED: Create generation pipeline only if not skipping generation
        if not skip_generation:
            print("[Load] Creating generation pipeline (SHARED TRANSFORMER) …")
            self.generation_pipeline = FluxPipeline(
                scheduler=base_components['scheduler'],
                text_encoder=base_components['text_encoder'],
                tokenizer=base_components['tokenizer'],
                text_encoder_2=base_components['text_encoder_2'],
                tokenizer_2=base_components['tokenizer_2'],
                vae=base_components['vae'],
                transformer=base_components['transformer'],  # Shared transformer
            ).to(self.device, self.torch_dtype)
            
            # Load IP-Adapter for generation pipeline if enabled
            if self.ip_adapter_enabled:
                self._setup_ip_adapter_for_pipeline(self.generation_pipeline)
        else:
            print("[Load] SKIPPING generation pipeline creation (skip_generation=True)")
        
        # MEMORY OPTIMIZED: Always create inpaint pipeline for direct inpainting support
        print("[Load] Creating inpaint pipeline (SHARED TRANSFORMER) …")
        self.inpaint_pipeline = FluxInpaintPipeline(
            scheduler=copy.deepcopy(base_components['scheduler']),
            text_encoder=base_components['text_encoder'],
            tokenizer=base_components['tokenizer'],
            text_encoder_2=base_components['text_encoder_2'],
            tokenizer_2=base_components['tokenizer_2'],
            vae=base_components['vae'],
            transformer=base_components['transformer'],  # Same shared transformer
        ).to(self.device, self.torch_dtype)
        
        # Pre-load all LoRA adapters (but don't activate them yet)
        self._preload_all_adapters(inference_lora_paths, inference_lora_strengths, skip_generation)
        
        # Setup detection helpers
        if self.face_inpainting:
            self._setup_face_detection()
        if self.upper_body_inpainting:
            self._setup_pose_detection()

        self.is_loaded = True
        print("[Load] Memory-optimized loading completed ✓")

    def _preload_all_adapters(self, inference_lora_paths, inference_lora_strengths, skip_generation=False):
        """Pre-load and FUSE all LoRA adapters following StableDiffusion approach."""
        print("[Load] Pre-loading and FUSING all LoRA adapters (StableDiffusion approach) …")
        
        # Prepare inference LoRAs
        inference_lora_paths = inference_lora_paths or []
        inference_lora_strengths = inference_lora_strengths or [1.0] * len(inference_lora_paths)
        if len(inference_lora_strengths) < len(inference_lora_paths):
            inference_lora_strengths.extend([1.0] * (len(inference_lora_paths) - len(inference_lora_strengths)))
        
        # FUSE generation LoRAs into base weights for better results (only if not skipping generation)
        if inference_lora_paths and not skip_generation:
            print(f"[Load] FUSING {len(inference_lora_paths)} generation LoRAs into base weights …")
            
            if self.low_vram:
                # Use StableDiffusion's sophisticated low VRAM approach
                self._fuse_loras_low_vram(inference_lora_paths, inference_lora_strengths)
            else:
                # Standard fusion approach
                self._fuse_loras_standard(inference_lora_paths, inference_lora_strengths)
        elif inference_lora_paths and skip_generation:
            print(f"[Load] SKIPPING generation LoRA fusion (skip_generation=True)")
            print(f"       → {len(inference_lora_paths)} inference LoRAs will be available for inpainting")
        
        # For face/upper body LoRAs, we'll load them on-demand during inpainting
        # This allows different LoRAs for different tasks while keeping main generation fused
        if self.face_lora_paths:
            print(f"[Load] Face LoRAs will be loaded on-demand during inpainting: {len(self.face_lora_paths)} LoRAs")
            for path, strength in zip(self.face_lora_paths, self.face_lora_strengths):
                print(f"       • {Path(path).name}  strength={strength}")
        
        if self.upper_body_lora_paths:
            print(f"[Load] Upper body LoRAs will be loaded on-demand during inpainting: {len(self.upper_body_lora_paths)} LoRAs")
            for path, strength in zip(self.upper_body_lora_paths, self.upper_body_lora_strengths):
                print(f"       • {Path(path).name}  strength={strength}")
        
        if not skip_generation:
            print("[Load] LoRA fusion completed - generation LoRAs are now part of base weights")
        else:
            print("[Load] LoRA preparation completed for inpainting-only mode")

    def _fuse_loras_standard(self, lora_paths: List[str], lora_strengths: List[float]):
        """Standard LoRA fusion approach (following StableDiffusion class)."""
        temp_pipeline = FluxPipeline(
            scheduler=None,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=None,
            transformer=self.generation_pipeline.transformer,  # Use the shared transformer
        )
        
        for i, (path, strength) in enumerate(zip(lora_paths, lora_strengths)):
            adapter_name = f"temp_lora_{i}"
            temp_pipeline.load_lora_weights(path, adapter_name=adapter_name)
            print(f"       • Loaded {Path(path).name} for fusion")
        
        # Set adapter strengths and fuse
        adapter_names = [f"temp_lora_{i}" for i in range(len(lora_paths))]
        temp_pipeline.set_adapters(adapter_names, lora_strengths)
        temp_pipeline.fuse_lora()
        
        # Clean up adapter metadata (weights are now fused)
        for adapter_name in adapter_names:
            temp_pipeline.unload_lora_weights(adapter_name)
        
        # FIXED: Ensure transformer is on correct device after fusion
        print("[Load] Moving transformer to device after LoRA fusion …")
        self.generation_pipeline.transformer = self.generation_pipeline.transformer.to(self.device, self.torch_dtype)
        self.inpaint_pipeline.transformer = self.inpaint_pipeline.transformer.to(self.device, self.torch_dtype)
        
        print(f"       → Fused {len(lora_paths)} LoRAs into transformer weights")
        print(f"       → Transformer moved to {self.device} with dtype {self.torch_dtype}")

    def _fuse_loras_low_vram(self, lora_paths: List[str], lora_strengths: List[float]):
        """Low VRAM LoRA fusion (following StableDiffusion class approach)."""
        print("[Load] Using low VRAM LoRA fusion approach …")
        
        # Create temporary pipeline for fusion
        temp_pipeline = FluxPipeline(
            scheduler=None,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=None,
            transformer=self.generation_pipeline.transformer,
        )
        
        for lora_path in lora_paths:
            lora_state_dict = load_file(lora_path)
            
            # Separate LoRA weights by transformer block type
            single_transformer_lora = {}
            double_transformer_lora = {}
            other_lora = {}
            
            single_block_key = "transformer.single_transformer_blocks."
            double_block_key = "transformer.transformer_blocks."
            
            for key, value in lora_state_dict.items():
                if single_block_key in key:
                    single_transformer_lora[key] = value
                elif double_block_key in key:
                    double_transformer_lora[key] = value
                else:
                    other_lora[key] = value
            
            # Process double blocks
            if double_transformer_lora:
                print(f"       • Processing double blocks from {Path(lora_path).name}")
                self.generation_pipeline.transformer.transformer_blocks = \
                    self.generation_pipeline.transformer.transformer_blocks.to(self.device)
                
                temp_pipeline.load_lora_weights(double_transformer_lora, adapter_name="temp_double")
                temp_pipeline.fuse_lora()
                temp_pipeline.unload_lora_weights("temp_double")
            
            # Process single blocks
            if single_transformer_lora:
                print(f"       • Processing single blocks from {Path(lora_path).name}")
                self.generation_pipeline.transformer.single_transformer_blocks = \
                    self.generation_pipeline.transformer.single_transformer_blocks.to(self.device)
                
                temp_pipeline.load_lora_weights(single_transformer_lora, adapter_name="temp_single")
                temp_pipeline.fuse_lora()
                temp_pipeline.unload_lora_weights("temp_single")
            
            # Process other LoRA weights
            if other_lora:
                print(f"       • Processing other weights from {Path(lora_path).name}")
                temp_pipeline.load_lora_weights(other_lora, adapter_name="temp_other")
                temp_pipeline.fuse_lora()
                temp_pipeline.unload_lora_weights("temp_other")
            
            # Cleanup
            del lora_state_dict
            del single_transformer_lora
            del double_transformer_lora
            del other_lora
            torch.cuda.empty_cache()
        
        # FIXED: Ensure entire transformer is on the correct device after fusion
        print("[Load] Moving complete transformer to device after LoRA fusion …")
        self.generation_pipeline.transformer = self.generation_pipeline.transformer.to(self.device, self.torch_dtype)
        self.inpaint_pipeline.transformer = self.inpaint_pipeline.transformer.to(self.device, self.torch_dtype)
        
        print(f"       → Low VRAM fusion completed for {len(lora_paths)} LoRAs")
        print(f"       → Transformer moved to {self.device} with dtype {self.torch_dtype}")

    def _switch_adapters(self, pipeline, adapter_set_name: str):
        """Load inpainting-specific LoRAs on demand (only for face/upper body)."""
        if adapter_set_name == "generation":
            print(f"[Adapter] Using fused generation LoRAs (already in base weights)")
            return
        elif adapter_set_name == "face":
            if self.face_lora_paths and not self.face_loras_loaded:
                print(f"[Adapter] Loading face LoRAs on-demand for inpainting …")
                for i, (path, strength) in enumerate(zip(self.face_lora_paths, self.face_lora_strengths)):
                    adapter_name = f"face_lora_{i}"
                    pipeline.load_lora_weights(path, adapter_name=adapter_name)
                    self.face_adapters.append((adapter_name, strength))
                    print(f"       • {Path(path).name}  strength={strength}")
                
                if self.face_adapters:
                    adapter_names = [name for name, strength in self.face_adapters]
                    adapter_strengths = [strength for name, strength in self.face_adapters]
                    pipeline.set_adapters(adapter_names, adapter_strengths)
                    self.face_loras_loaded = True
                    
        elif adapter_set_name == "upper_body":
            if self.upper_body_lora_paths and not self.upper_body_loras_loaded:
                print(f"[Adapter] Loading upper body LoRAs on-demand for inpainting …")
                for i, (path, strength) in enumerate(zip(self.upper_body_lora_paths, self.upper_body_lora_strengths)):
                    adapter_name = f"upper_body_lora_{i}"
                    pipeline.load_lora_weights(path, adapter_name=adapter_name)
                    self.upper_body_adapters.append((adapter_name, strength))
                    print(f"       • {Path(path).name}  strength={strength}")
                
                if self.upper_body_adapters:
                    adapter_names = [name for name, strength in self.upper_body_adapters]
                    adapter_strengths = [strength for name, strength in self.upper_body_adapters]
                    pipeline.set_adapters(adapter_names, adapter_strengths)
                    self.upper_body_loras_loaded = True

    def cleanup_adapters(self) -> None:
        """Clean up only the inpainting LoRA adapters (generation LoRAs are fused)."""
        print("[Cleanup] Cleaning up inpainting LoRA adapters …")
        
        # Only clean up face and upper body adapters (generation LoRAs are fused)
        face_and_upper_adapters = self.face_adapters + self.upper_body_adapters
        
        for adapter_name, _ in face_and_upper_adapters:
            try:
                if self.inpaint_pipeline:
                    self.inpaint_pipeline.unload_lora_weights(adapter_name)
                    print(f"       • Unloaded {adapter_name}")
            except Exception as e:
                print(f"[Cleanup] Warning: Could not unload {adapter_name}: {e}")
        
        self.face_adapters.clear()
        self.upper_body_adapters.clear()
        self.face_loras_loaded = False
        self.upper_body_loras_loaded = False
        
        print("[Cleanup] Inpainting adapter cleanup completed")
        print("[Cleanup] Note: Generation LoRAs remain fused in base weights")

    def _setup_ip_adapter_for_pipeline(self, pipeline) -> None:
        """Setup IP-Adapter for a specific pipeline."""
        try:
            print("[IP-Adapter] Loading IP-Adapter for pipeline …")
            
            pipeline.load_ip_adapter(
                self.ip_adapter_model_path,
                weight_name="ip_adapter.safetensors"
            )
            
            if hasattr(pipeline, 'set_ip_adapter_scale'):
                default_scale = self.ip_adapter_scales[0] if self.ip_adapter_scales else 0.6
                pipeline.set_ip_adapter_scale(default_scale)
                print(f"[IP-Adapter] Set default scale to {default_scale}")
            
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
                print("[IP-Adapter] Warning: No valid reference images found")
                self.ip_adapter_enabled = False
                return
                
            print(f"[IP-Adapter] Successfully loaded {len(self.preprocessed_ip_images)} reference images")
            
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
                
                # RELAXED: Lower confidence threshold for LoRA-generated faces
                if confidence < 0.2:  # Reduced from 0.3 to 0.2
                    print(f"      ↳ Skipped: Low confidence ({confidence:.3f} < 0.2)")
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
                
                # RELAXED: More permissive face size ratios
                min_face_ratio = 0.0005  # Reduced from 0.001 to 0.0005
                max_face_ratio = 0.25    # Increased from 0.15 to 0.25
                
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
        
        results = self.pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        
        if (left_shoulder.visibility < 0.3 and right_shoulder.visibility < 0.3):
            return None
        
        # Calculate face + neck region
        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)
        
        face_landmarks = [landmarks[NOSE], landmarks[LEFT_EYE], landmarks[RIGHT_EYE]]
        face_points = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks if lm.visibility > 0.3]
        
        if not face_points:
            return None
            
        face_xs = [p[0] for p in face_points]
        face_ys = [p[1] for p in face_points]
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
        
        if w * h < width * height * 0.01:
            return None
        
        return (x, y, w, h)

    # ------------------------------ NEW: Direct inpainting methods -------- #

    def process_image_directly(
        self,
        image_path: str,
        output_filename: Optional[str] = None,
        output_ext: str = ".png",
        face_prompt_override: Optional[str] = None,
        upper_body_prompt_override: Optional[str] = None,
    ) -> Image.Image:
        """Process an existing image directly with face/upper body inpainting."""
        if not self.is_loaded:
            # Load model for inpainting only
            self.load_model(skip_generation=True)

        if not self.inpaint_pipeline:
            raise RuntimeError("Inpainting pipeline not loaded")

        if not (self.face_inpainting or self.upper_body_inpainting):
            print("[Direct] No inpainting options enabled, returning original image")
            return Image.open(image_path).convert("RGB")

        # Load the input image
        try:
            img = Image.open(image_path).convert("RGB")
            print(f"[Direct] Loaded image: {image_path} -> {img.size}")
        except Exception as e:
            print(f"[Direct] Error loading image {image_path}: {e}")
            raise

        processed_img = img.copy()

        # Override prompts if provided
        original_face_prompt = self.face_prompt
        original_upper_body_prompt = self.upper_body_prompt
        
        if face_prompt_override:
            self.face_prompt = face_prompt_override
            print(f"[Direct] Using face prompt override: {face_prompt_override}")
        
        if upper_body_prompt_override:
            self.upper_body_prompt = upper_body_prompt_override
            print(f"[Direct] Using upper body prompt override: {upper_body_prompt_override}")

        try:
            # Upper body inpainting (face + neck region) with adapter switching
            if self.upper_body_inpainting:
                upper_body_region = self.detect_upper_body_region(processed_img)
                if upper_body_region:
                    print(f"[Direct] Face+neck region detected, switching to upper body adapters …")
                    self._switch_adapters(self.inpaint_pipeline, "upper_body")
                    processed_img = self._inpaint_upper_body(processed_img, upper_body_region)
                else:
                    print(f"[Direct] No face+neck region detected, skipping upper body inpainting")

            # Face inpainting (fine details) with adapter switching
            if self.face_inpainting:
                faces = self.detect_faces(processed_img)
                if faces:
                    print(f"[Direct] {len(faces)} face(s) found, switching to face adapters …")
                    self._switch_adapters(self.inpaint_pipeline, "face")
                    for b in faces:
                        processed_img = self._inpaint_face(processed_img, b)
                else:
                    print(f"[Direct] No faces found, skipping face inpainting")

            # Save the processed image
            if output_filename:
                if not os.path.splitext(output_filename)[1]:
                    output_filename += output_ext
            else:
                input_stem = Path(image_path).stem
                output_filename = f"{input_stem}_processed{output_ext}"
            
            output_path = self.output_dir / output_filename
            processed_img.save(output_path)
            print(f"[Direct] ✔ Processed image saved → {output_path.relative_to(self.output_dir.parent)}")

        finally:
            # Restore original prompts
            self.face_prompt = original_face_prompt
            self.upper_body_prompt = original_upper_body_prompt

        return processed_img

    # ------------------------------ inpainting ---------------------------- #

    def _inpaint_face(self, img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
        """Face inpainting using shared pipeline with adapter switching."""
        if not self.inpaint_pipeline:
            print("[Face] No inpainting pipeline available")
            return img
            
        x, y, w, h = box
        try:
            print(f"      ↳ Processing face at ({x}, {y}) size {w}x{h}")
            
            # Switch to face adapters only if we have them
            if self.face_adapters:
                self._switch_adapters(self.inpaint_pipeline, "face")
                print(f"      ↳ Using face-specific adapters")
            else:
                print(f"      ↳ Using current adapters (no face-specific adapters available)")
            
            # Calculate processing size
            min_size, max_size = 512, 1024
            face_size = max(w, h)
            if face_size < min_size:
                scale_factor = min_size / face_size
            elif face_size > max_size:
                scale_factor = max_size / face_size
            else:
                scale_factor = 1.0
                
            target_w = ((int(w * scale_factor) + 7) // 8) * 8
            target_h = ((int(h * scale_factor) + 7) // 8) * 8
            
            # Extract and prepare face region
            padding = 0.08
            pad_x, pad_y = int(w * padding), int(h * padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.width, x + w + pad_x)
            y2 = min(img.height, y + h + pad_y)
            
            face_region = img.crop((x1, y1, x2, y2))
            face_resized = face_region.resize((target_w, target_h), Image.LANCZOS)
            
            # Create inpainting mask
            mask_padding = 0.12
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            mask_x1 = int(target_w * mask_padding)
            mask_y1 = int(target_h * mask_padding)
            mask_x2 = int(target_w * (1 - mask_padding))
            mask_y2 = int(target_h * (1 - mask_padding))
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
            
            blur_size = max(5, min(target_w, target_h) // 20)
            if blur_size % 2 == 0:
                blur_size += 1
            mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            mask_pil = Image.fromarray(mask_blurred)
            
            # Enhanced prompts
            enhanced_prompt = f"{self.face_prompt}, natural facial proportions, preserve face shape"
            enhanced_negative = f"{self.face_neg_prompt}, oversized features, enlarged face, distorted proportions"
            
            print(f"      ↳ Running face inpainting at {target_w}x{target_h}...")
            
            # Use shared inpainting pipeline with current adapters
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=face_resized,
                mask_image=mask_pil,
                num_inference_steps=32,
                guidance_scale=self.face_guidance_scale,
                strength=0.75,
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
            
            inpainted_face = result.images[0]
            face_final = inpainted_face.resize(face_region.size, Image.LANCZOS)
            
            # Blend back into original image
            result_img = img.copy()
            original_region_img = img.crop((x1, y1, x2, y2))
            
            # Create blend mask
            final_mask = np.zeros((face_region.size[1], face_region.size[0]), dtype=np.uint8)
            face_x_in_region, face_y_in_region = pad_x, pad_y
            feather = min(w, h) // 8
            
            # Tighter mask bounds to prevent expansion
            mask_x1 = max(0, face_x_in_region + int(feather * 1.5))
            mask_y1 = max(0, face_y_in_region + int(feather * 1.5))
            mask_x2 = min(face_region.size[0], face_x_in_region + w - int(feather * 1.5))
            mask_y2 = min(face_region.size[1], face_y_in_region + h - int(feather * 1.5))
            
            if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                final_mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
                final_blur_size = max(5, feather)
                if final_blur_size % 2 == 0:
                    final_blur_size += 1
                final_mask_blurred = cv2.GaussianBlur(final_mask, (final_blur_size, final_blur_size), 0)
            else:
                final_mask_blurred = np.ones((face_region.size[1], face_region.size[0]), dtype=np.uint8) * 128
            
            # Blend images
            original_np = np.array(original_region_img).astype(np.float32)
            final_np = np.array(face_final).astype(np.float32)
            blend_mask_np = final_mask_blurred.astype(np.float32) / 255.0
            
            if len(blend_mask_np.shape) == 2:
                blend_mask_np = np.expand_dims(blend_mask_np, axis=2)
            
            # Scale down the blend mask to preserve more original
            blend_mask_np = blend_mask_np * 0.95
            
            blended_np = original_np * (1 - blend_mask_np) + final_np * blend_mask_np
            blended_img = Image.fromarray(np.uint8(blended_np))
            
            result_img.paste(blended_img, (x1, y1))
            
            print(f"      ✓ Face inpainting completed")
            return result_img
            
        except Exception as e:
            print(f"[Face] Inpainting failed: {e}")
            traceback.print_exc()
            return img

    def _inpaint_upper_body(self, img: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
        """Upper body inpainting using shared pipeline with adapter switching."""
        if not self.inpaint_pipeline:
            print("[Upper Body] No inpainting pipeline available")
            return img
            
        x, y, w, h = region
        try:
            print(f"      ↳ Processing face+neck region at ({x}, {y}) size {w}x{h} with upper body adapters")
            
            # Switch to upper body adapters
            self._switch_adapters(self.inpaint_pipeline, "upper_body")
            
            # Calculate processing size
            min_size, max_size = 512, 1024
            region_size = max(w, h)
            if region_size < min_size:
                scale_factor = min_size / region_size
            elif region_size > max_size:
                scale_factor = max_size / region_size
            else:
                scale_factor = 1.0
                
            target_w = ((int(w * scale_factor) + 7) // 8) * 8
            target_h = ((int(h * scale_factor) + 7) // 8) * 8
            
            # Extract and prepare region
            face_neck_region = img.crop((x, y, x + w, y + h))
            region_resized = face_neck_region.resize((target_w, target_h), Image.LANCZOS)
            
            # Create inpainting mask
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
            
            # Apply edge falloff
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
            
            # Use shared inpainting pipeline with upper body adapters
            result = self.inpaint_pipeline(
                prompt=self.upper_body_prompt,
                negative_prompt=self.upper_body_neg_prompt,
                image=region_resized,
                mask_image=mask_pil,
                num_inference_steps=35,
                guidance_scale=self.upper_body_guidance_scale,
                strength=0.8,
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
            
            inpainted_region = result.images[0]
            region_final = inpainted_region.resize(face_neck_region.size, Image.LANCZOS)
            
            # Blend back into original image
            result_img = img.copy()
            original_region_img = img.crop((x, y, x + w, y + h))
            
            # Create blend mask with falloff
            blend_mask = np.ones((face_neck_region.size[1], face_neck_region.size[0]), dtype=np.uint8) * 255
            falloff_ratio = 0.08
            falloff_x = max(1, int(face_neck_region.size[0] * falloff_ratio))
            falloff_y = max(1, int(face_neck_region.size[1] * falloff_ratio))
            
            for i in range(falloff_x):
                alpha = i / falloff_x
                blend_mask[:, i] = blend_mask[:, i] * alpha
                blend_mask[:, -(i+1)] = blend_mask[:, -(i+1)] * alpha
            
            for i in range(falloff_y):
                alpha = i / falloff_y
                blend_mask[i, :] = blend_mask[i, :] * alpha
                blend_mask[-(i+1), :] = blend_mask[-(i+1), :] * alpha
            
            blend_blur_size = max(3, min(face_neck_region.size) // 30)
            if blend_blur_size % 2 == 0:
                blend_blur_size += 1
            blend_mask_blurred = cv2.GaussianBlur(blend_mask, (blend_blur_size, blend_blur_size), 0)
            
            # Blend images
            original_np = np.array(original_region_img).astype(np.float32)
            final_np = np.array(region_final).astype(np.float32)
            blend_mask_np = blend_mask_blurred.astype(np.float32) / 255.0
            
            if len(blend_mask_np.shape) == 2:
                blend_mask_np = np.expand_dims(blend_mask_np, axis=2)
            
            blended_np = original_np * (1 - blend_mask_np) + final_np * blend_mask_np
            blended_img = Image.fromarray(np.uint8(blended_np))
            
            result_img.paste(blended_img, (x, y))
            
            print(f"      ✓ Face+neck inpainting completed with upper body adapters")
            return result_img
            
        except Exception as e:
            print(f"[Upper Body] Inpainting failed: {e}")
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
        # IP-Adapter parameters for per-generation control
        ip_adapter_images: Optional[List[Union[str, Image.Image]]] = None,
        ip_adapter_scales: Optional[List[float]] = None,
        # NEW: Direct inpainting parameters
        skip_generation: bool = False,
        start_image: Optional[str] = None,
    ) -> List[Image.Image]:
        """Generate images with memory-optimized adapter management or process existing images directly."""
        if not self.is_loaded:
            self.load_model(skip_generation=skip_generation)

        # NEW: Direct inpainting mode
        if skip_generation and start_image:
            print(f"[Gen] DIRECT INPAINTING MODE: Processing existing image {start_image}")
            if not os.path.exists(start_image):
                raise FileNotFoundError(f"Start image not found: {start_image}")
            
            # Process the image directly with inpainting
            processed_img = self.process_image_directly(
                image_path=start_image,
                output_filename=output_filenames[0] if output_filenames else None,
                output_ext=output_ext,
                # Use first prompt as face prompt override if provided
                face_prompt_override=prompts[0] if prompts else None,
            )
            
            print("[Gen] Direct inpainting completed ✓")
            return [processed_img]

        # Original generation mode
        if not self.generation_pipeline:
            raise RuntimeError("Generation pipeline not loaded")

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
            
            # Switch to generation adapters
            self._switch_adapters(self.generation_pipeline, "generation")
            
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
            
            # Add IP-Adapter parameters if available
            if generation_ip_images and self.ip_adapter_enabled:
                gen_kwargs["ip_adapter_image"] = generation_ip_images
                
                if generation_ip_scales:
                    try:
                        if hasattr(self.generation_pipeline, 'set_ip_adapter_scale'):
                            scale_to_set = generation_ip_scales[0] if len(generation_ip_scales) == 1 else generation_ip_scales
                            self.generation_pipeline.set_ip_adapter_scale(scale_to_set)
                            print(f"[IP-Adapter] Set scale to {scale_to_set}")
                        else:
                            gen_kwargs["ip_adapter_scale"] = generation_ip_scales[0] if len(generation_ip_scales) == 1 else generation_ip_scales
                    except Exception as e:
                        print(f"[IP-Adapter] Warning: Could not set scale: {e}")
            
            try:
                print(f"      → Generating with generation adapters...")
                res = self.generation_pipeline(**gen_kwargs)
                images = res.images
            except Exception as e:
                print(f"[Gen] Error during generation: {e}")
                print("[Gen] Trying without IP-Adapter...")
                # Fallback without IP-Adapter
                gen_kwargs.pop("ip_adapter_image", None)
                gen_kwargs.pop("ip_adapter_scale", None)
                res = self.generation_pipeline(**gen_kwargs)
                images = res.images
            
            # Process each generated image with adapter switching
            for j, im in enumerate(images):
                processed_img = im
                
                # Upper body inpainting (face + neck region) with adapter switching
                if self.upper_body_inpainting:
                    upper_body_region = self.detect_upper_body_region(processed_img)
                    if upper_body_region:
                        print(f"      ↳ Face+neck region detected, switching to upper body adapters …")
                        # Only switch adapters if we have upper body-specific ones, otherwise use current adapters
                        if self.upper_body_adapters:
                            self._switch_adapters(self.inpaint_pipeline, "upper_body")
                        else:
                            print(f"      ↳ No upper body-specific adapters, using current adapters")
                        processed_img = self._inpaint_upper_body(processed_img, upper_body_region)
                    else:
                        print(f"      ↳ No face+neck region detected, skipping upper body inpainting")
                
                # Face inpainting (fine details) with adapter switching
                if self.face_inpainting:
                    faces = self.detect_faces(processed_img)
                    if faces:
                        print(f"      ↳ {len(faces)} face(s) found, switching to face adapters …")
                        # Only switch adapters if we have face-specific ones, otherwise use current adapters
                        if self.face_adapters:
                            self._switch_adapters(self.inpaint_pipeline, "face")
                        else:
                            print(f"      ↳ No face-specific adapters, using current adapters")
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

        return all_imgs

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup_adapters()
        except:
            pass  # Ignore cleanup errors during destruction


# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory-optimized Flux image generation with adapter switching and direct inpainting")
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
    ip_adapter_cfg = cfg.get("ip_adapter", {})
    
    # NEW: Check for direct inpainting mode
    skip_generation = generate_cfg.get("skip_generation", False)
    start_image = generate_cfg.get("start_image")
    
    if skip_generation and not start_image:
        raise SystemExit("✖ skip_generation=true requires start_image to be specified")
    
    if skip_generation and start_image and not os.path.exists(start_image):
        raise SystemExit(f"✖ Start image not found: {start_image}")
    
    # Prompts ---------------------------------------------------------------
    prompts = generate_cfg.get("prompts", [])
    if not prompts and not skip_generation:
        raise SystemExit("✖ No prompts provided in config file")
    
    # For direct inpainting, prompts are optional (used as face prompt override)
    if skip_generation and not prompts:
        prompts = [""]  # Empty prompt as placeholder

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
        # IP-Adapter configuration
        ip_adapter_model_path=ip_adapter_cfg.get("model_path", "XLabs-AI/flux-ip-adapter"),
        ip_adapter_images=ip_adapter_cfg.get("reference_images", []),
        ip_adapter_scales=ip_adapter_cfg.get("scales", []),
        ip_adapter_target_size=tuple(ip_adapter_cfg.get("target_size", [512, 512])),
        low_vram=model_cfg.get("low_vram", False),
        config=cfg,
    )

    try:
        # Check if we need to enable inpainting for direct mode
        if skip_generation and not (generate_cfg.get("face_inpainting", False) or generate_cfg.get("upper_body_inpainting", False)):
            print("[Main] WARNING: Direct inpainting mode but no inpainting options enabled!")
            print("[Main] Consider enabling face_inpainting or upper_body_inpainting in config")

        # Load weights ----------------------------------------------------------
        flux.load_model(
            inference_lora_paths=model_cfg.get("inference_lora_paths", []),
            inference_lora_strengths=model_cfg.get("inference_lora_strengths", []),
            skip_generation=skip_generation,
        )

        # Run generation or direct inpainting -----------------------------------
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
            # Per-generation IP-Adapter control
            ip_adapter_images=generate_cfg.get("ip_adapter_images"),
            ip_adapter_scales=generate_cfg.get("ip_adapter_scales"),
            # NEW: Direct inpainting parameters
            skip_generation=skip_generation,
            start_image=start_image,
        )
    finally:
        # Always cleanup adapters
        flux.cleanup_adapters()


if __name__ == "__main__":
    main()