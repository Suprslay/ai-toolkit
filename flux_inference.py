#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flux Inference and Inpainting Script
-----------------------------------
A single‑file utility for running image generation with Flux models.
Features
~~~~~~~~
* LoRA loading (multiple adapters, independent strengths)
* Optional face detection + in‑place inpainting with a second Flux pipeline
* NEW: Upper body inpainting (face + neck only)
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
from typing import Dict, List, Optional, Tuple

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
    CLIPTextModel,
    CLIPTokenizer,
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


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class FluxInference:
    """Light‑weight manager around Diffusers' `FluxPipeline`."""

    # ------------------------------ init & helpers -------------------------- #

    def __init__(
        self,
        model_path: str,
        output_dir: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        face_inpainting: bool = False,
        upper_body_inpainting: bool = False,  # NEW
        face_lora_paths: Optional[List[str]] = None,
        face_lora_strengths: Optional[List[float]] = None,
        upper_body_lora_paths: Optional[List[str]] = None,  # NEW
        upper_body_lora_strengths: Optional[List[float]] = None,  # NEW
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
        self.face_prompt = "FSLAYMNSHA BSLAYMNSHA a woman, smooth skin, make-up, model, fashion"
        self.face_neg_prompt = (
            "deformed face, distorted face, disfigured, low res, bad anatomy, cartoon, anime, painting, sketch, low quality, pixelated, jpeg artifacts, oversaturated"
        )
        self.face_guidance_scale = 3.5  # Lower guidance for Flux
        self.face_expansion = 0
        self.face_detection_model = "mediapipe"
        self.save_face_debug = True

        # NEW: Upper body‑related opts
        # UPDATED: "Upper body" now means FACE + NECK only (not shoulders/chest)
        # This is more targeted than full upper torso but larger than face-only
        self.upper_body_inpainting = upper_body_inpainting
        self.upper_body_lora_paths = upper_body_lora_paths or []
        self.upper_body_lora_strengths = upper_body_lora_strengths or [0.75] * len(self.upper_body_lora_paths)
        if len(self.upper_body_lora_strengths) < len(self.upper_body_lora_paths):
            self.upper_body_lora_strengths.extend(
                [0.75] * (len(self.upper_body_lora_paths) - len(self.upper_body_lora_strengths))
            )
        # Prompt optimized for face + neck region
        self.upper_body_prompt = "FSLAYMNSHA BSLAYMNSHA a woman, beautiful face, elegant neck, portrait, smooth skin, professional photography, high quality"
        self.upper_body_neg_prompt = (
            "deformed face, distorted face, disfigured, low res, bad anatomy, cartoon, anime, painting, sketch, low quality, pixelated, jpeg artifacts, oversaturated"
        )
        self.upper_body_guidance_scale = 3.5
        self.save_upper_body_debug = True

        # Internal state
        self.pipeline: FluxPipeline | None = None
        self.inpaint_pipeline: FluxInpaintPipeline | None = None
        self.face_detector = None
        self.pose_detector = None  # NEW
        self.is_loaded = False
        self.face_loras_loaded = False  # Track if face LoRAs are loaded
        self.upper_body_loras_loaded = False  # NEW: Track if upper body LoRAs are loaded

        print(f"[Init] model_path={model_path}  device={device}  dtype={dtype}")
        if face_inpainting:
            print("[Init] Face inpainting ENABLED")
            if self.face_lora_paths:
                for p, s in zip(self.face_lora_paths, self.face_lora_strengths):
                    print(f"       • {Path(p).name}  strength={s} (will load on-demand)")
        
        if upper_body_inpainting:
            print("[Init] Upper body inpainting ENABLED")
            print("       → Upper body = FACE + NECK region only")
            print("       → Does NOT include shoulders, chest, or torso") 
            print("       → More targeted than full torso, larger than face-only")
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

        self.pipeline = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder2,
            tokenizer_2=tokenizer2,
            vae=vae,
            transformer=transformer,
        ).to(self.device, self.torch_dtype)

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
            
        # DON'T initialize inpaint pipeline here to avoid style leak
        # It will be initialized on-demand when first detection is made

        self.is_loaded = True
        print("[Load] Done ✓")

    # ------------------------------ detection utils ------------------------ #

    def _setup_face_detection(self) -> None:
        """Enhanced face detection setup with better settings for full-body images."""
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1=full range (up to 5m) - better for full body shots
                min_detection_confidence=0.2  # Lowered from 0.5 to 0.2 for better small face detection
            )
            print(f"[Face] MediaPipe face detection model loaded (full range, confidence=0.2)")
        except ImportError:
            print(f"[Face] ERROR: MediaPipe not installed. Please install with 'pip install mediapipe'")
            raise ValueError("MediaPipe not installed. Please install with 'pip install mediapipe'")

    def _setup_pose_detection(self) -> None:
        """NEW: Setup pose detection for shoulder detection."""
        try:
            import mediapipe as mp
            if not hasattr(self, 'mp'):
                self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,  # Highest accuracy
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print(f"[Pose] MediaPipe pose detection model loaded (complexity=2, confidence=0.3)")
        except ImportError:
            print(f"[Pose] ERROR: MediaPipe not installed. Please install with 'pip install mediapipe'")
            raise ValueError("MediaPipe not installed. Please install with 'pip install mediapipe'")

    def detect_faces(self, img: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Improved face detection optimized for full-body images where faces are smaller.
        """
        if not self.face_detector:
            self._setup_face_detection()
        
        # Convert image to RGB for MediaPipe (MediaPipe expects RGB)
        img_rgb = np.array(img)
        height, width, _ = img_rgb.shape
        
        print(f"[Face Detection] Image size: {width}x{height}")
        
        # Process the image with MediaPipe
        results = self.face_detector.process(img_rgb)
        
        boxes: List[Tuple[int, int, int, int]] = []
        if results.detections:
            print(f"[Face Detection] Found {len(results.detections)} raw detections")
            
            for i, detection in enumerate(results.detections):
                # Get confidence score
                confidence = detection.score[0] if detection.score else 0
                print(f"[Face Detection] Detection {i+1}: confidence={confidence:.3f}")
                
                # LOWERED confidence threshold for full-body images where faces are smaller/more distant
                if confidence < 0.3:  # Was 0.7, now 0.3 for better detection of distant faces
                    print(f"      ↳ Skipped: Low confidence ({confidence:.3f} < 0.3)")
                    continue
                    
                # Get bounding box from detection
                bbox = detection.location_data.relative_bounding_box
                
                # Convert normalized coordinates to pixel values
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                print(f"      ↳ Raw bbox: x={x}, y={y}, w={w}, h={h}")
                
                # Filter out faces that are too small or too large
                face_area = w * h
                image_area = width * height
                face_ratio = face_area / image_area
                
                print(f"      ↳ Face ratio: {face_ratio:.4f} ({face_area}/{image_area})")
                
                # RELAXED size filtering for full-body shots
                # Full body images typically have faces that are 0.1% to 15% of image area
                min_face_ratio = 0.001  # Was 0.01, now 0.001 (0.1% instead of 1%)
                max_face_ratio = 0.15   # Was 0.5, now 0.15 (15% instead of 50%)
                
                if face_ratio < min_face_ratio:
                    print(f"      ↳ Skipped: Face too small ({face_ratio:.4f} < {min_face_ratio})")
                    continue
                if face_ratio > max_face_ratio:
                    print(f"      ↳ Skipped: Face too large ({face_ratio:.4f} > {max_face_ratio})")
                    continue
                
                # RELAXED edge detection - allow faces closer to edges in full-body shots
                edge_threshold = 2  # Was 5, now 2 pixels
                near_edge = (x < edge_threshold or y < edge_threshold or 
                            x + w > width - edge_threshold or y + h > height - edge_threshold)
                
                if near_edge:
                    print(f"      ↳ Warning: Face near edge, but allowing for full-body shots")
                    # Don't skip - just warn. Full body shots often have faces near top edge
                
                # Apply minimal expansion - just enough for context
                exp = 0.05  # 5% expansion
                nx = max(0, int(x - w * exp))
                ny = max(0, int(y - h * exp))
                nw = min(width - nx, int(w * (1 + 2 * exp)))
                nh = min(height - ny, int(h * (1 + 2 * exp)))
                
                print(f"      ↳ Final bbox: x={nx}, y={ny}, w={nw}, h={nh}")
                boxes.append((nx, ny, nw, nh))
        else:
            print("[Face Detection] No detections found by MediaPipe")
        
        # Sort by face size (largest first) to prioritize main subjects
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        
        print(f"[Face Detection] Final result: {len(boxes)} faces passed filtering")
        return boxes

    def detect_upper_body_region(self, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        NEW: Detect the face + neck region using pose detection.
        
        UPDATED DEFINITION: "Upper body" now means FACE + NECK only:
        - Face region (detected via pose landmarks for head position)
        - Neck area (from face to shoulder line)
        - Does NOT include shoulders, chest, or torso
        
        This provides more targeted enhancement than full upper torso inpainting
        while still being larger than face-only inpainting.
        
        Use cases:
        - Portrait neck/face enhancement
        - Targeted head and neck style transfer
        - Face + neck skin improvement  
        - Comprehensive facial region processing
        
        Returns (x, y, w, h) for the face+neck region to inpaint, or None if no pose detected.
        """
        if not self.pose_detector:
            self._setup_pose_detection()
        
        # Convert image to RGB for MediaPipe
        img_rgb = np.array(img)
        height, width, _ = img_rgb.shape
        
        print(f"[Upper Body Detection] Image size: {width}x{height}")
        
        # Process the image with MediaPipe Pose
        results = self.pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            print("[Upper Body Detection] No pose landmarks detected")
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Key landmarks for shoulder detection (MediaPipe pose landmark indices)
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 7
        RIGHT_EAR = 8
        
        # Get shoulder positions
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        
        # Check if shoulders are visible (confidence > threshold)
        shoulder_confidence_threshold = 0.3
        if (left_shoulder.visibility < shoulder_confidence_threshold and 
            right_shoulder.visibility < shoulder_confidence_threshold):
            print(f"[Upper Body Detection] Shoulders not visible enough (L:{left_shoulder.visibility:.2f}, R:{right_shoulder.visibility:.2f})")
            return None
        
        # Convert normalized coordinates to pixel coordinates
        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)
        
        # Get key facial landmarks for better region definition
        nose = landmarks[NOSE]
        left_eye = landmarks[LEFT_EYE] 
        right_eye = landmarks[RIGHT_EYE]
        
        # Calculate face region bounds using facial landmarks
        face_landmarks = [nose, left_eye, right_eye]
        face_points = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks if lm.visibility > 0.3]
        
        if not face_points:
            print("[Upper Body Detection] Insufficient facial landmarks visible")
            return None
            
        # Get face bounding box
        face_xs = [p[0] for p in face_points]
        face_ys = [p[1] for p in face_points]
        face_center_x = sum(face_xs) // len(face_xs)
        face_top_y = min(face_ys)
        
        print(f"[Upper Body Detection] Face center: ({face_center_x}, {face_top_y})")
        print(f"[Upper Body Detection] Left shoulder: ({left_shoulder_x}, {left_shoulder_y})")
        print(f"[Upper Body Detection] Right shoulder: ({right_shoulder_x}, {right_shoulder_y})")
        
        # Calculate the top of the region (above the face for forehead/hair)
        face_height_estimate = max(left_shoulder_y, right_shoulder_y) - face_top_y
        top_padding = int(face_height_estimate * 0.3)  # 30% above face for forehead/hair
        region_top_y = max(0, face_top_y - top_padding)
        
        # Calculate neck bottom boundary (just at shoulder line, not below)
        # Remove the shoulder padding since we only want face + neck
        neck_bottom_y = max(left_shoulder_y, right_shoulder_y)
        
        print(f"[Upper Body Detection] Region bounds: top={region_top_y}, bottom={neck_bottom_y}")
        
        # Determine the width of the face + neck region
        # Use shoulder width as reference but make it more conservative for face+neck
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)
        center_x = (left_shoulder_x + right_shoulder_x) // 2
        
        # Use a smaller extension factor since we're only doing face+neck
        extension_factor = 0.3  # Reduced from 0.5 to 0.3 for more targeted region
        region_width = int(shoulder_width * (1 + 2 * extension_factor))
        
        # Calculate the bounding box for the face + neck region  
        x = max(0, center_x - region_width // 2)
        y = region_top_y
        w = min(width - x, region_width)
        h = min(height - y, neck_bottom_y - region_top_y)
        
        # Ensure we have a reasonable region size
        min_region_ratio = 0.01  # Reduced from 0.1 since face+neck is smaller
        region_area = w * h
        image_area = width * height
        
        if region_area < image_area * min_region_ratio:
            print(f"[Upper Body Detection] Face+neck region too small ({region_area}/{image_area} = {region_area/image_area:.3f})")
            return None
        
        print(f"[Upper Body Detection] Face+neck region: x={x}, y={y}, w={w}, h={h}")
        print(f"[Upper Body Detection] Region covers {region_area/image_area:.1%} of image")
        
        return (x, y, w, h)

    # ------------------------------ inpainting ---------------------------- #

    def _init_inpaint_pipeline(self) -> None:
        """Initialize inpaint pipeline on-demand to avoid style leak."""
        if self.inpaint_pipeline is not None:
            return
        print("[Inpaint] Building inpaint pipeline (on-demand) …")
        
        # Instead of loading from pretrained, reuse components from main pipeline
        if self.pipeline is not None:
            self.inpaint_pipeline = FluxInpaintPipeline(
                vae=self.pipeline.vae,
                transformer=self.pipeline.transformer,
                text_encoder=self.pipeline.text_encoder,
                text_encoder_2=self.pipeline.text_encoder_2,
                tokenizer=self.pipeline.tokenizer,
                tokenizer_2=self.pipeline.tokenizer_2,
                scheduler=self.pipeline.scheduler
            ).to(self.device)
        else:
            # Fallback to loading from pretrained if main pipeline isn't loaded yet
            self.inpaint_pipeline = FluxInpaintPipeline.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype
            ).to(self.device)

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
        """NEW: Load upper body LoRAs into inpaint pipeline on-demand to prevent style leak."""
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
        """
        Clean, effective face inpainting approach for Flux models.
        """
        x, y, w, h = box
        try:
            # Initialize inpaint pipeline and load face LoRAs on-demand
            self._init_inpaint_pipeline()
            self._load_face_loras()  # Load face LoRAs only when needed
            
            # Create debug directory if needed
            if self.save_face_debug:
                debug_dir = self.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
            
            print(f"      ↳ Processing face at ({x}, {y}) size {w}x{h}")
            
            # 1. Determine optimal processing size
            # Flux works best at certain resolutions, typically 512px minimum for good quality
            min_size = 512
            max_size = 1024  # Keep reasonable to avoid VRAM issues
            
            # Calculate target size based on face dimensions
            face_size = max(w, h)
            if face_size < min_size:
                scale_factor = min_size / face_size
            elif face_size > max_size:
                scale_factor = max_size / face_size
            else:
                scale_factor = 1.0
                
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Ensure dimensions are multiples of 8 (common requirement for diffusion models)
            target_w = ((target_w + 7) // 8) * 8
            target_h = ((target_h + 7) // 8) * 8
            
            print(f"      ↳ Target processing size: {target_w}x{target_h} (scale: {scale_factor:.2f})")
            
            # 2. Extract face region with minimal padding for context
            padding = 0.15  # 15% padding for context
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            # Calculate padded region bounds
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.width, x + w + pad_x)
            y2 = min(img.height, y + h + pad_y)
            
            # Extract the region
            face_region = img.crop((x1, y1, x2, y2))
            original_region_size = face_region.size
            
            # Save original for debug
            if self.save_face_debug:
                face_path = debug_dir / f"face_original_{timestamp}.png"
                face_region.save(face_path)
                print(f"      ↳ Debug: Original face region saved")
            
            # 3. Resize to target processing size
            face_resized = face_region.resize((target_w, target_h), Image.LANCZOS)
            
            # 4. Create a clean, simple mask
            # Focus on the core facial features area
            mask_padding = 0.05  # 10% padding from edges for the mask
            mask_x1 = int(target_w * mask_padding)
            mask_y1 = int(target_h * mask_padding)
            mask_x2 = int(target_w * (1 - mask_padding))
            mask_y2 = int(target_h * (1 - mask_padding))
            
            # Create base mask
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
            
            # Apply moderate gaussian blur for smooth edges
            blur_size = max(5, min(target_w, target_h) // 20)  # Adaptive blur size
            if blur_size % 2 == 0:
                blur_size += 1  # Ensure odd number for gaussian blur
                
            mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            mask_pil = Image.fromarray(mask_blurred)
            
            # Save mask for debug
            if self.save_face_debug:
                mask_path = debug_dir / f"mask_{timestamp}.png"
                mask_pil.save(mask_path)
                print(f"      ↳ Debug: Mask saved")
            
            # 5. Enhanced prompt for better face generation
            enhanced_prompt = f"{self.face_prompt}"
            enhanced_negative = f"{self.face_neg_prompt}"
            
            print(f"      ↳ Running face inpainting at {target_w}x{target_h}...")
            
            # 6. Run inpainting with optimal settings for Flux
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=face_resized,
                mask_image=mask_pil,
                num_inference_steps=32,  # Slightly more steps for better quality
                guidance_scale=self.face_guidance_scale,  # Lower guidance for Flux (3.5)
                strength=0.85,           # Moderate strength to preserve some original features
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
            
            inpainted_face = result.images[0]
            
            # Save inpainted result for debug
            if self.save_face_debug:
                inpainted_path = debug_dir / f"face_inpainted_{timestamp}.png"
                inpainted_face.save(inpainted_path)
                print(f"      ↳ Debug: Inpainted face saved")
            
            # 7. Resize back to original region size
            face_final = inpainted_face.resize(original_region_size, Image.LANCZOS)
            
            # 8. Create final blending mask for pasting back
            # This mask should be slightly smaller to avoid harsh edges
            final_mask = np.zeros((original_region_size[1], original_region_size[0]), dtype=np.uint8)
            
            # Calculate the actual face position within the padded region
            face_x_in_region = pad_x
            face_y_in_region = pad_y
            
            # Create mask for the face area with some feathering
            feather = min(w, h) // 10  # 10% of face size for feathering
            
            mask_x1 = max(0, face_x_in_region + feather)
            mask_y1 = max(0, face_y_in_region + feather)
            mask_x2 = min(original_region_size[0], face_x_in_region + w - feather)
            mask_y2 = min(original_region_size[1], face_y_in_region + h - feather)
            
            if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                final_mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
                
                # Apply blur for smooth blending
                final_blur_size = max(3, feather // 2)
                if final_blur_size % 2 == 0:
                    final_blur_size += 1
                    
                final_mask_blurred = cv2.GaussianBlur(final_mask, (final_blur_size, final_blur_size), 0)
            else:
                # Fallback: use the entire region
                final_mask_blurred = np.ones((original_region_size[1], original_region_size[0]), dtype=np.uint8) * 128
            
            # 9. Blend and paste back into original image
            result_img = img.copy()
            original_region_img = img.crop((x1, y1, x2, y2))
            
            # Convert to numpy for blending
            original_np = np.array(original_region_img).astype(np.float32)
            final_np = np.array(face_final).astype(np.float32)
            blend_mask_np = final_mask_blurred.astype(np.float32) / 255.0
            
            # Expand mask dimensions for RGB
            if len(blend_mask_np.shape) == 2:
                blend_mask_np = np.expand_dims(blend_mask_np, axis=2)
            
            # Blend
            blended_np = original_np * (1 - blend_mask_np) + final_np * blend_mask_np
            blended_img = Image.fromarray(np.uint8(blended_np))
            
            # Paste back into result
            result_img.paste(blended_img, (x1, y1))
            
            # Save final result for debug
            if self.save_face_debug:
                final_path = debug_dir / f"face_final_{timestamp}.png"
                result_img.crop((x1, y1, x2, y2)).save(final_path)
                print(f"      ↳ Debug: Final result saved")
            
            print(f"      ✓ Face inpainting completed successfully")
            return result_img
            
        except Exception as e:
            print(f"[Face] Inpainting failed: {e}")
            traceback.print_exc()
            return img

    def _inpaint_upper_body(self, img: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
        """
        NEW: Inpaint the face + neck region.
        
        UPDATED: "Upper body" now means FACE + NECK only:
        - Face area (including forehead, cheeks, chin)
        - Neck area (from jawline to shoulder line)
        - Does NOT include shoulders, chest, or torso
        
        This provides targeted enhancement for the most important portrait features
        while being more focused than full upper torso inpainting.
        
        Enhanced approach optimized for face+neck regions and maintaining coherence.
        """
        x, y, w, h = region
        try:
            # Initialize inpaint pipeline and load upper body LoRAs on-demand
            self._init_inpaint_pipeline()
            self._load_upper_body_loras()  # Load upper body LoRAs only when needed
            
            # Create debug directory if needed
            if self.save_upper_body_debug:
                debug_dir = self.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
            
            print(f"      ↳ Processing face+neck region at ({x}, {y}) size {w}x{h}")
            
            # 1. Determine optimal processing size for face+neck region
            # Face+neck regions are medium-sized, so we use balanced sizing
            min_size = 512  # Good minimum for face+neck quality
            max_size = 1024  # Reasonable maximum to avoid VRAM issues
            
            # Calculate scaling based on the larger dimension
            region_size = max(w, h)
            if region_size < min_size:
                scale_factor = min_size / region_size
            elif region_size > max_size:
                scale_factor = max_size / region_size
            else:
                scale_factor = 1.0
                
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Ensure dimensions are multiples of 8 for good stability
            target_w = ((target_w + 7) // 8) * 8
            target_h = ((target_h + 7) // 8) * 8
            
            print(f"      ↳ Target processing size: {target_w}x{target_h} (scale: {scale_factor:.2f})")
            
            # 2. Extract face+neck region 
            face_neck_region = img.crop((x, y, x + w, y + h))
            original_region_size = face_neck_region.size
            
            # Save original for debug
            if self.save_upper_body_debug:
                original_path = debug_dir / f"face_neck_original_{timestamp}.png"
                face_neck_region.save(original_path)
                print(f"      ↳ Debug: Original face+neck region saved")
            
            # 3. Resize to target processing size
            region_resized = face_neck_region.resize((target_w, target_h), Image.LANCZOS)
            
            # 4. Create optimized mask for face+neck inpainting
            # More conservative masking since this is a more targeted region
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Create a mask that covers most of the face+neck but preserves some edges
            edge_margin = 0.08  # 8% margin from edges (more conservative)
            center_x1 = int(target_w * edge_margin)
            center_y1 = int(target_h * edge_margin)
            center_x2 = int(target_w * (1 - edge_margin))
            center_y2 = int(target_h * (1 - edge_margin))
            
            # Create base mask
            mask[center_y1:center_y2, center_x1:center_x2] = 255
            
            # Apply gaussian blur for smooth transitions
            # Use moderate blur for face+neck region
            blur_size = max(9, min(target_w, target_h) // 25)  # Moderate blur
            if blur_size % 2 == 0:
                blur_size += 1
                
            mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
            # Create gentle edge falloff for natural blending
            edge_falloff = 0.12  # 12% edge falloff
            falloff_pixels = int(min(target_w, target_h) * edge_falloff)
            
            # Apply edge falloff
            for i in range(falloff_pixels):
                alpha = i / falloff_pixels
                
                # Top edge
                if i < target_h:
                    mask_blurred[i, :] = mask_blurred[i, :] * alpha
                
                # Bottom edge  
                if target_h - 1 - i >= 0:
                    mask_blurred[target_h - 1 - i, :] = mask_blurred[target_h - 1 - i, :] * alpha
                
                # Left edge
                if i < target_w:
                    mask_blurred[:, i] = mask_blurred[:, i] * alpha
                
                # Right edge
                if target_w - 1 - i >= 0:
                    mask_blurred[:, target_w - 1 - i] = mask_blurred[:, target_w - 1 - i] * alpha
            
            mask_pil = Image.fromarray(mask_blurred)
            
            # Save mask for debug
            if self.save_upper_body_debug:
                mask_path = debug_dir / f"face_neck_mask_{timestamp}.png"
                mask_pil.save(mask_path)
                print(f"      ↳ Debug: Face+neck mask saved")
            
            # 5. Enhanced prompt for face+neck generation
            enhanced_prompt = f"{self.upper_body_prompt}"
            enhanced_negative = f"{self.upper_body_neg_prompt}"
            
            print(f"      ↳ Running face+neck inpainting at {target_w}x{target_h}...")
            
            # 6. Run inpainting with settings optimized for face+neck regions
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=enhanced_negative,
                image=region_resized,
                mask_image=mask_pil,
                num_inference_steps=35,  # Good balance for face+neck complexity
                guidance_scale=self.upper_body_guidance_scale,  # 3.5 for Flux
                strength=0.8,            # Good strength for face+neck enhancement
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
            
            inpainted_face_neck = result.images[0]
            
            # Save inpainted result for debug
            if self.save_upper_body_debug:
                inpainted_path = debug_dir / f"face_neck_inpainted_{timestamp}.png"
                inpainted_face_neck.save(inpainted_path)
                print(f"      ↳ Debug: Inpainted face+neck saved")
            
            # 7. Resize back to original region size
            face_neck_final = inpainted_face_neck.resize(original_region_size, Image.LANCZOS)
            
            # 8. Create blending mask for pasting back
            # For face+neck, we want smooth but more controlled blending
            blend_mask = np.ones((original_region_size[1], original_region_size[0]), dtype=np.uint8) * 255
            
            # Create edge falloff for smooth blending
            falloff_ratio = 0.08  # 8% falloff at edges (more conservative for face+neck)
            falloff_x = max(1, int(original_region_size[0] * falloff_ratio))
            falloff_y = max(1, int(original_region_size[1] * falloff_ratio))
            
            # Apply gradient falloff
            for i in range(falloff_x):
                alpha = i / falloff_x
                blend_mask[:, i] = blend_mask[:, i] * alpha
                blend_mask[:, -(i+1)] = blend_mask[:, -(i+1)] * alpha
            
            for i in range(falloff_y):
                alpha = i / falloff_y
                blend_mask[i, :] = blend_mask[i, :] * alpha
                blend_mask[-(i+1), :] = blend_mask[-(i+1), :] * alpha
            
            # Apply moderate gaussian blur for smooth blending
            blend_blur_size = max(3, min(original_region_size) // 30)  # Smaller blur for face+neck
            if blend_blur_size % 2 == 0:
                blend_blur_size += 1
            
            blend_mask_blurred = cv2.GaussianBlur(blend_mask, (blend_blur_size, blend_blur_size), 0)
            
            # 9. Blend and paste back into original image
            result_img = img.copy()
            original_region_img = img.crop((x, y, x + w, y + h))
            
            # Convert to numpy for blending
            original_np = np.array(original_region_img).astype(np.float32)
            final_np = np.array(face_neck_final).astype(np.float32)
            blend_mask_np = blend_mask_blurred.astype(np.float32) / 255.0
            
            # Expand mask dimensions for RGB
            if len(blend_mask_np.shape) == 2:
                blend_mask_np = np.expand_dims(blend_mask_np, axis=2)
            
            # Blend with face+neck result
            blended_np = original_np * (1 - blend_mask_np) + final_np * blend_mask_np
            blended_img = Image.fromarray(np.uint8(blended_np))
            
            # Paste back into result
            result_img.paste(blended_img, (x, y))
            
            # Save final result for debug
            if self.save_upper_body_debug:
                final_path = debug_dir / f"face_neck_final_{timestamp}.png"
                result_img.crop((x, y, x + w, y + h)).save(final_path)
                print(f"      ↳ Debug: Final face+neck result saved")
            
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
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        seed: int = -1,
        output_filenames: Optional[List[str]] = None,
        output_ext: str = ".png",
    ) -> List[Image.Image]:
        if not self.is_loaded:
            self.load_model()

        all_imgs: List[Image.Image] = []
        for idx, prompt in enumerate(prompts):
            use_seed = random.randint(0, 2**32 - 1) if seed < 0 else seed
            print(f"[Gen] Prompt {idx+1}/{len(prompts)}  seed={use_seed}\n      → {prompt}")
            gen = torch.Generator(device=self.device).manual_seed(use_seed)
            res = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=gen,
            )
            images = res.images
            
            # Save intermediate images before inpainting if debug is enabled
            if self.save_face_debug or self.save_upper_body_debug:
                debug_dir = self.output_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
                for i, img in enumerate(images):
                    debug_path = debug_dir / f"initial_generation_{timestamp}_{i}.png"
                    img.save(debug_path)
                    print(f"[Debug] Saved initial generation image {i+1}/{len(images)} to {debug_path}")
            
            # Process each generated image
            for j, im in enumerate(images):
                processed_img = im
                
                # Upper body inpainting (do this first for broader changes)
                # NOTE: Upper body = FACE + NECK only (not shoulders/chest/torso)
                # This provides targeted enhancement of the most important portrait features
                if self.upper_body_inpainting:
                    upper_body_region = self.detect_upper_body_region(processed_img)
                    if upper_body_region:
                        print(f"      ↳ Face+neck region detected, inpainting …")
                        processed_img = self._inpaint_upper_body(processed_img, upper_body_region)
                    else:
                        print(f"      ↳ No face+neck region detected, skipping upper body inpainting")
                
                # Face inpainting (do this after upper body for fine details)
                # NOTE: Face inpainting is a smaller, more targeted operation for facial features only
                # Note: Face LoRAs and Upper Body LoRAs are loaded independently to prevent conflicts
                if self.face_inpainting:
                    # Reset LoRA state to ensure clean face inpainting
                    if self.upper_body_loras_loaded:
                        self.upper_body_loras_loaded = False  # Force reload of face LoRAs
                    
                    faces = self.detect_faces(processed_img)
                    if faces:
                        print(f"      ↳ {len(faces)} face(s) found, inpainting …")
                        for b in faces:
                            processed_img = self._inpaint_face(processed_img, b)
                    else:
                        print(f"      ↳ No faces found, skipping face inpainting")
                
                images[j] = processed_img

            # Save final images ---------------------------------------------------
            for j, im in enumerate(images):
                if output_filenames and idx * num_images_per_prompt + j < len(output_filenames):
                    fname = output_filenames[idx * num_images_per_prompt + j]
                    if not os.path.splitext(fname)[1]:
                        fname += output_ext
                else:
                    slug = _slugify(prompt)
                    # Add index number to filename (1-based for user-friendliness)
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
    p = argparse.ArgumentParser(description="Flux image generation helper with upper body inpainting")
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
        upper_body_inpainting=generate_cfg.get("upper_body_inpainting", False),  # NEW
        face_lora_paths=generate_cfg.get("face_lora_paths", []),
        face_lora_strengths=generate_cfg.get("face_lora_strengths", []),
        upper_body_lora_paths=generate_cfg.get("upper_body_lora_paths", []),  # NEW
        upper_body_lora_strengths=generate_cfg.get("upper_body_lora_strengths", []),  # NEW
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
        guidance_scale=generate_cfg.get("guidance_scale", 7.0),
        num_images_per_prompt=generate_cfg.get("num_images", 1),
        seed=generate_cfg.get("seed", -1),
        output_ext=generate_cfg.get("ext", ".png"),
    )


if __name__ == "__main__":
    main()