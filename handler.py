"""RunPod Serverless Handler for Flux Training and Inference with Simplified LoRA Support"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from typing import Union, OrderedDict, List, Dict, Any
from dotenv import load_dotenv
import runpod
import base64
import requests
from PIL import Image
import io
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import yaml
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# Import rembg for background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("rembg imported successfully - background removal enabled")
except ImportError as e:
    print(f"Warning: Could not import rembg: {e}")
    print("Background removal will be disabled. Install rembg with: pip install rembg")
    REMBG_AVAILABLE = False

# Load the .env file if it exists
load_dotenv()
sys.path.insert(0, os.getcwd())

# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc
# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

# Import FluxInference class from the flux_inference.py
try:
    # Add the current directory to Python path to import flux_inference
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from flux_inference import FluxInference
    FLUX_INFERENCE_AVAILABLE = True
    print_acc("FluxInference imported successfully")
except ImportError as e:
    print_acc(f"Warning: Could not import FluxInference: {e}")
    print_acc("Inference functionality will be disabled")
    FLUX_INFERENCE_AVAILABLE = False

# Initialize S3 client globally
s3_client = None

# Initialize rembg session globally for reuse
rembg_session = None

def initialize_rembg():
    """Initialize rembg session for background removal"""
    global rembg_session
    if not REMBG_AVAILABLE:
        print_acc("rembg not available - background removal disabled")
        return False
    
    try:
        # Initialize with u2net model (good balance of speed and quality)
        # You can change to 'u2netp' for faster processing or 'silueta' for better quality
        rembg_session = new_session('u2net')
        print_acc("rembg session initialized successfully with u2net model")
        return True
    except Exception as e:
        print_acc(f"Warning: Could not initialize rembg session: {e}")
        print_acc("Background removal will be disabled")
        return False

def remove_background(image):
    """
    Remove background from PIL Image using rembg
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image object with background removed (RGBA format)
    """
    if not REMBG_AVAILABLE or rembg_session is None:
        print_acc("Background removal not available - returning original image")
        return image
    
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        # Save as PNG to preserve any existing alpha channel
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Remove background
        print_acc("Removing background...")
        output = remove(img_byte_arr.getvalue(), session=rembg_session)
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(output))
        
        # Ensure RGBA format for transparency
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        print_acc("Background removed successfully")
        return result_image
        
    except Exception as e:
        print_acc(f"Error removing background: {e}")
        print_acc("Returning original image")
        return image

def initialize_s3_client():
    """Initialize S3 client with credentials from environment variables"""
    global s3_client
    try:
        # Try to initialize S3 client
        # Credentials can come from:
        # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # 2. IAM roles (if running on EC2)
        # 3. AWS credentials file
        s3_client = boto3.client('s3')
        print_acc("S3 client initialized successfully")
        return True
    except Exception as e:
        print_acc(f"Warning: Could not initialize S3 client: {e}")
        print_acc("S3 functionality will be disabled. Set AWS credentials to enable S3 support.")
        return False

def handle_start_image_s3(config_data, temp_dir):
    """
    Handle start_image S3 paths by downloading them to local temp directory
    
    Args:
        config_data: Configuration dictionary
        temp_dir: Temporary directory for files
    
    Returns:
        dict: Updated configuration with local start_image paths
    """
    updated_config = config_data.copy()
    generate_cfg = updated_config.get("generate", {})
    
    start_image = generate_cfg.get("start_image")
    
    if start_image and isinstance(start_image, str) and start_image.startswith('s3://'):
        try:
            print_acc(f"Downloading start_image from S3: {start_image}")
            
            # Download the S3 image
            image = download_from_s3(start_image)
            
            # Create a filename for the start image
            # Extract filename from S3 path or create a default one
            s3_filename = os.path.basename(start_image.rstrip('/'))
            if not s3_filename or not any(s3_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']):
                s3_filename = "start_image.png"
            
            # Save to temp directory
            local_start_image_path = os.path.join(temp_dir, s3_filename)
            
            # Convert to RGB if necessary and save
            if image.mode == 'RGBA':
                # Keep as PNG to preserve transparency
                image.save(local_start_image_path, "PNG")
            else:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # Save as original format or PNG
                if s3_filename.lower().endswith('.png'):
                    image.save(local_start_image_path, "PNG")
                elif s3_filename.lower().endswith(('.jpg', '.jpeg')):
                    image.save(local_start_image_path, "JPEG")
                else:
                    image.save(local_start_image_path, "PNG")
            
            # Update config with local path
            generate_cfg["start_image"] = local_start_image_path
            
            print_acc(f"Downloaded start_image to: {local_start_image_path}")
            
        except Exception as e:
            error_msg = f"Failed to download start_image from S3: {str(e)}"
            print_acc(error_msg)
            raise Exception(error_msg)
    
    return updated_config

def parse_lora_path(path):
    """
    Parse LoRA path and determine if it's local or needs to be downloaded from HF
    
    Supports only:
    - repo_id/filename.safetensors (e.g., "anuraj-sisyphus/avatar-loras/slaymnsha_face_v1.safetensors")
    - local file paths (e.g., "/path/to/lora.safetensors" or "./lora.safetensors")
    
    Args:
        path: LoRA path string
        
    Returns:
        dict: {"type": "hf_file|local", "repo_id": str, "filename": str, "local_path": str}
    """
    if not isinstance(path, str):
        raise ValueError("LoRA path must be a string")
    
    # Check if it's a local path (exists on filesystem or has local path indicators)
    if (os.path.exists(path) or 
        path.startswith('/') or 
        path.startswith('./') or 
        path.startswith('../') or
        (len(path.split('/')) == 1 and '.' in path)):  # single filename with extension
        return {
            "type": "local",
            "repo_id": None,
            "filename": None,
            "local_path": path
        }
    
    # Parse HF short format: repo_id/filename.safetensors
    path_parts = path.split('/')
    
    if len(path_parts) >= 3 and path_parts[-1].endswith('.safetensors'):
        # Format: user/repo/filename.safetensors or user/repo/subdir/filename.safetensors
        filename = path_parts[-1]
        repo_id = '/'.join(path_parts[:-1])
        
        return {
            "type": "hf_file",
            "repo_id": repo_id,
            "filename": filename,
            "local_path": None
        }
    
    # If we can't parse it as HF format, treat as local
    return {
        "type": "local",
        "repo_id": None,
        "filename": None,
        "local_path": path
    }

def download_hf_lora(repo_id, filename, cache_dir="./lora_cache"):
    """
    Download specific LoRA file from Hugging Face
    
    Args:
        repo_id: HF repository ID (e.g., "anuraj-sisyphus/avatar-loras")
        filename: Specific file to download (e.g., "slaymnsha_face_v1.safetensors")
        cache_dir: Local cache directory
        
    Returns:
        str: Path to downloaded file
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        print_acc(f"Downloading LoRA: {repo_id}/{filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            token=os.getenv("HF_TOKEN")  # Optional HF token for private repos
        )
        print_acc(f"Downloaded LoRA to: {local_path}")
        return local_path
        
    except Exception as e:
        raise Exception(f"Failed to download LoRA {repo_id}/{filename}: {str(e)}")

def resolve_lora_paths(lora_paths, cache_dir="./lora_cache"):
    """
    Resolve LoRA paths, downloading from HF if necessary
    
    Args:
        lora_paths: List of LoRA paths
        cache_dir: Cache directory for downloads
        
    Returns:
        list: List of resolved local file paths
    """
    if not lora_paths:
        return []
    
    resolved_paths = []
    
    for lora_path in lora_paths:
        try:
            parsed = parse_lora_path(lora_path)
            
            if parsed["type"] == "local":
                # Local file - verify it exists
                if os.path.exists(parsed["local_path"]):
                    resolved_paths.append(parsed["local_path"])
                    print_acc(f"Using local LoRA: {parsed['local_path']}")
                else:
                    raise Exception(f"Local LoRA file not found: {parsed['local_path']}")
                    
            elif parsed["type"] == "hf_file":
                # Download specific HF file
                downloaded_path = download_hf_lora(
                    parsed["repo_id"], 
                    parsed["filename"], 
                    cache_dir
                )
                resolved_paths.append(downloaded_path)
                
        except Exception as e:
            error_msg = f"Error resolving LoRA path '{lora_path}': {str(e)}"
            print_acc(error_msg)
            raise Exception(error_msg)
    
    print_acc(f"Resolved {len(resolved_paths)} LoRA files")
    return resolved_paths

def update_config_with_resolved_loras(config_data, cache_dir="./lora_cache"):
    """
    Update configuration with resolved LoRA paths
    
    Args:
        config_data: Configuration dictionary
        cache_dir: Cache directory for downloads
        
    Returns:
        dict: Updated configuration with local LoRA paths
    """
    updated_config = config_data.copy()
    
    # Handle model-level LoRAs for inference
    model_cfg = updated_config.get("model", {})
    if "inference_lora_paths" in model_cfg:
        model_cfg["inference_lora_paths"] = resolve_lora_paths(
            model_cfg["inference_lora_paths"], cache_dir
        )
    
    # Handle generate-level LoRAs
    generate_cfg = updated_config.get("generate", {})
    
    # Face LoRAs
    if "face_lora_paths" in generate_cfg:
        generate_cfg["face_lora_paths"] = resolve_lora_paths(
            generate_cfg["face_lora_paths"], cache_dir
        )
    
    # Upper body LoRAs  
    if "upper_body_lora_paths" in generate_cfg:
        generate_cfg["upper_body_lora_paths"] = resolve_lora_paths(
            generate_cfg["upper_body_lora_paths"], cache_dir
        )
    
    return updated_config

def download_s3_folder_to_dataset(bucket_name, folder_path="", remove_bg=True):
    """
    Download all files from S3 folder to datasets/input, preserving folder structure
    
    Args:
        bucket_name: S3 bucket name
        folder_path: Folder path within bucket (optional)
        remove_bg: Whether to remove background from images
    
    Returns:
        str: Path to the dataset folder
    """
    if s3_client is None:
        raise Exception("S3 client not initialized. Check AWS credentials.")
    
    # Create dataset directory
    dataset_path = "datasets/input"
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        # Ensure folder path ends with / if not empty
        if folder_path and not folder_path.endswith('/'):
            folder_path += '/'
        
        print_acc(f"Downloading all files from s3://{bucket_name}/{folder_path}")
        
        # List all objects in the bucket/folder
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_path)
        
        file_count = 0
        processed_count = 0
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip directories (keys ending with /)
                    if key.endswith('/'):
                        continue
                    
                    # Get filename from the key
                    filename = os.path.basename(key)
                    if not filename:  # Skip if no filename
                        continue
                    
                    # Download the file
                    local_path = os.path.join(dataset_path, filename)
                    
                    print_acc(f"Downloading: {key} -> {filename}")
                    s3_client.download_file(bucket_name, key, local_path)
                    file_count += 1
                    
                    # Check if it's an image file and process background removal
                    if remove_bg and is_image_file(filename):
                        try:
                            print_acc(f"Processing image for background removal: {filename}")
                            image = Image.open(local_path)
                            processed_image = remove_background(image)
                            
                            # Save as PNG to preserve transparency
                            processed_path = os.path.splitext(local_path)[0] + '.png'
                            processed_image.save(processed_path, "PNG")
                            
                            # Remove original if format changed
                            if processed_path != local_path:
                                os.remove(local_path)
                                print_acc(f"Converted and processed: {filename} -> {os.path.basename(processed_path)}")
                            else:
                                print_acc(f"Processed: {filename}")
                            
                            processed_count += 1
                            
                        except Exception as e:
                            print_acc(f"Error processing image {filename}: {e}")
                            print_acc(f"Keeping original image: {filename}")
        
        if remove_bg and REMBG_AVAILABLE:
            print_acc(f"Successfully downloaded {file_count} files to {dataset_path}")
            print_acc(f"Background removed from {processed_count} images")
        else:
            print_acc(f"Successfully downloaded {file_count} files to {dataset_path}")
            if remove_bg and not REMBG_AVAILABLE:
                print_acc("Background removal was requested but rembg is not available")
        
        return dataset_path
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            raise Exception(f"S3 bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            raise Exception(f"Access denied to S3 bucket '{bucket_name}'")
        else:
            raise Exception(f"S3 error: {e}")
    except Exception as e:
        raise Exception(f"Failed to download from S3: {str(e)}")

def is_image_file(filename):
    """Check if filename is an image file based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_extensions

def download_from_s3(s3_url_or_object):
    """
    Download image from S3 bucket
    
    Args:
        s3_url_or_object: S3 URL string or object with bucket/key info
    
    Returns:
        PIL Image object
    """
    if s3_client is None:
        raise Exception("S3 client not initialized. Check AWS credentials.")
    
    try:
        bucket_name = None
        object_key = None
        
        if isinstance(s3_url_or_object, str):
            # Parse S3 URL: s3://bucket-name/path/to/image.jpg
            if s3_url_or_object.startswith('s3://'):
                url_parts = s3_url_or_object[5:].split('/', 1)
                bucket_name = url_parts[0]
                object_key = url_parts[1] if len(url_parts) > 1 else ''
            else:
                raise ValueError("S3 URL must start with 's3://'")
                
        elif isinstance(s3_url_or_object, dict):
            # Object format: {"bucket": "my-bucket", "key": "path/to/image.jpg"}
            bucket_name = s3_url_or_object.get('bucket')
            object_key = s3_url_or_object.get('key')
            
            if not bucket_name or not object_key:
                raise ValueError("S3 object must have 'bucket' and 'key' fields")
        else:
            raise ValueError("S3 input must be URL string or object with bucket/key")
        
        # Download the object
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()
        
        # Create PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        return image
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            raise Exception(f"S3 bucket '{bucket_name}' does not exist")
        elif error_code == 'NoSuchKey':
            raise Exception(f"S3 object '{object_key}' not found in bucket '{bucket_name}'")
        elif error_code == 'AccessDenied':
            raise Exception(f"Access denied to S3 object s3://{bucket_name}/{object_key}")
        else:
            raise Exception(f"S3 error: {e}")
    except Exception as e:
        raise Exception(f"Failed to download from S3: {str(e)}")

def save_images_to_dataset(images_input, remove_bg=True):
    """
    Save files to datasets/input folder with optional background removal
    For S3 folders: Download everything as-is
    For individual files: Keep original behavior but add background removal
    
    Args:
        images_input: Can be:
            - List of image data (base64 strings, URLs, S3 URLs, or objects)
            - S3 folder object: {"s3_folder": {"bucket": "name", "folder": "path"}}
            - S3 folder URL: "s3://bucket-name/folder-path/"
        remove_bg: Whether to remove background from images (default: True)
    
    Returns:
        str: Path to the created dataset folder
    """
    # Create dataset directory
    dataset_path = "datasets/input"
    os.makedirs(dataset_path, exist_ok=True)
    
    # Handle S3 folder input - download everything
    if isinstance(images_input, dict) and 's3_folder' in images_input:
        s3_folder = images_input['s3_folder']
        bucket_name = s3_folder.get('bucket')
        folder_path = s3_folder.get('folder', '')
        
        if not bucket_name:
            raise Exception("S3 folder object must have 'bucket' field")
        
        return download_s3_folder_to_dataset(bucket_name, folder_path, remove_bg)
        
    elif isinstance(images_input, str) and images_input.startswith('s3://') and images_input.endswith('/'):
        # S3 folder URL format: "s3://bucket-name/folder-path/"
        url_parts = images_input[5:].rstrip('/').split('/', 1)
        bucket_name = url_parts[0]
        folder_path = url_parts[1] if len(url_parts) > 1 else ''
        
        return download_s3_folder_to_dataset(bucket_name, folder_path, remove_bg)
        
    elif isinstance(images_input, list):
        # Original behavior for individual files - process each image
        print_acc(f"Processing {len(images_input)} individual images")
        
        processed_count = 0
        for i, image_data in enumerate(images_input, 1):
            try:
                image = None
                
                # Handle different image input formats
                if isinstance(image_data, str):
                    if image_data.startswith('s3://'):
                        # S3 URL (single file)
                        image = download_from_s3(image_data)
                    elif image_data.startswith('http'):
                        # Download from HTTP/HTTPS URL
                        response = requests.get(image_data)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                    elif image_data.startswith('data:image'):
                        # Data URL format: data:image/png;base64,iVBORw0KGgoAAAANSU...
                        header, data = image_data.split(',', 1)
                        image_bytes = base64.b64decode(data)
                        image = Image.open(io.BytesIO(image_bytes))
                    else:
                        # Assume base64 string
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        
                elif isinstance(image_data, dict):
                    if 'bucket' in image_data and 'key' in image_data:
                        # S3 object format: {"bucket": "my-bucket", "key": "path/to/image.jpg"}
                        image = download_from_s3(image_data)
                    elif 'data' in image_data:
                        # Base64 object format: {"data": "base64string", "filename": "optional"}
                        image_bytes = base64.b64decode(image_data['data'])
                        image = Image.open(io.BytesIO(image_bytes))
                    else:
                        raise ValueError(f"Unsupported object format for image {i}")
                else:
                    raise ValueError(f"Unsupported image format for image {i}")
                
                # Apply background removal if requested
                if remove_bg:
                    print_acc(f"Processing image {i} for background removal...")
                    image = remove_background(image)
                    processed_count += 1
                    
                    # Save as PNG to preserve transparency
                    image_path = os.path.join(dataset_path, f"{i}.png")
                    image.save(image_path, "PNG")
                    print_acc(f"Saved processed image {i} to {image_path}")
                else:
                    # Convert to RGB if necessary and save as PNG (original behavior)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_path = os.path.join(dataset_path, f"{i}.png")
                    image.save(image_path, "PNG")
                    print_acc(f"Saved image {i} to {image_path}")
                
            except Exception as e:
                error_msg = f"Error processing image {i}: {str(e)}"
                print_acc(error_msg)
                raise Exception(error_msg)
                
        if remove_bg and REMBG_AVAILABLE:
            print_acc(f"Successfully processed {len(images_input)} images to {dataset_path}")
            print_acc(f"Background removed from {processed_count} images")
        else:
            print_acc(f"Successfully processed {len(images_input)} images to {dataset_path}")
            if remove_bg and not REMBG_AVAILABLE:
                print_acc("Background removal was requested but rembg is not available")
    else:
        raise Exception("Invalid images input format")
    
    return dataset_path

def save_images_for_inference(images_input, temp_dir: str, remove_bg=False) -> List[str]:
    """
    Save images for inference and return list of file paths.
    
    Args:
        images_input: Same format as save_images_to_dataset
        temp_dir: Temporary directory to save images
        remove_bg: Whether to remove background (default: False for inference)
    
    Returns:
        List of file paths to saved images
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    images = []
    
    # Handle S3 folder input
    if isinstance(images_input, dict) and 's3_folder' in images_input:
        s3_folder = images_input['s3_folder']
        bucket_name = s3_folder.get('bucket')
        folder_path = s3_folder.get('folder', '')
        
        if not bucket_name:
            raise Exception("S3 folder object must have 'bucket' field")
        
        images = download_s3_folder_images(bucket_name, folder_path)
        
    elif isinstance(images_input, str) and images_input.startswith('s3://') and images_input.endswith('/'):
        url_parts = images_input[5:].rstrip('/').split('/', 1)
        bucket_name = url_parts[0]
        folder_path = url_parts[1] if len(url_parts) > 1 else ''
        
        images = download_s3_folder_images(bucket_name, folder_path)
        
    elif isinstance(images_input, list):
        for i, image_data in enumerate(images_input, 1):
            image = None
            
            if isinstance(image_data, str):
                if image_data.startswith('s3://'):
                    image = download_from_s3(image_data)
                elif image_data.startswith('http'):
                    response = requests.get(image_data)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                elif image_data.startswith('data:image'):
                    header, data = image_data.split(',', 1)
                    image_bytes = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
            elif isinstance(image_data, dict):
                if 'bucket' in image_data and 'key' in image_data:
                    image = download_from_s3(image_data)
                elif 'data' in image_data:
                    image_bytes = base64.b64decode(image_data['data'])
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    raise ValueError(f"Unsupported object format for image {i}")
            else:
                raise ValueError(f"Unsupported image format for image {i}")
            
            images.append(image)
    else:
        raise Exception("Invalid images input format")
    
    # Save images and return paths
    image_paths = []
    processed_count = 0
    for i, image in enumerate(images, 1):
        # Apply background removal if requested
        if remove_bg:
            print_acc(f"Processing reference image {i} for background removal...")
            image = remove_background(image)
            processed_count += 1
            
            # Save as PNG to preserve transparency
            image_path = os.path.join(temp_dir, f"ref_image_{i}.png")
            image.save(image_path, "PNG")
            print_acc(f"Saved processed reference image {i} to {image_path}")
        else:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_path = os.path.join(temp_dir, f"ref_image_{i}.png")
            image.save(image_path, "PNG")
            print_acc(f"Saved reference image {i} to {image_path}")
        
        image_paths.append(image_path)
    
    if remove_bg and REMBG_AVAILABLE and processed_count > 0:
        print_acc(f"Background removed from {processed_count} reference images")
    
    return image_paths

def download_s3_folder_images(bucket_name, folder_path=""):
    """
    Download all images from an S3 folder (for inference only)
    """
    if s3_client is None:
        raise Exception("S3 client not initialized. Check AWS credentials.")
    
    try:
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Ensure folder path ends with / if not empty
        if folder_path and not folder_path.endswith('/'):
            folder_path += '/'
        
        print_acc(f"Listing images in s3://{bucket_name}/{folder_path}")
        
        # List objects in the bucket/folder
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_path)
        
        image_keys = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip directories (keys ending with /)
                    if key.endswith('/'):
                        continue
                    
                    # Check if file has image extension
                    file_ext = os.path.splitext(key.lower())[1]
                    if file_ext in image_extensions:
                        image_keys.append(key)
        
        if not image_keys:
            raise Exception(f"No images found in s3://{bucket_name}/{folder_path}")
        
        images = []
        print_acc(f"Downloading {len(image_keys)} images from S3...")
        
        for i, key in enumerate(image_keys, 1):
            try:
                print_acc(f"Downloading image {i}/{len(image_keys)}: {key}")
                image = download_from_s3({"bucket": bucket_name, "key": key})
                images.append(image)
            except Exception as e:
                print_acc(f"Error downloading {key}: {str(e)}")
                raise Exception(f"Failed to download {key}: {str(e)}")
        
        print_acc(f"Successfully downloaded {len(images)} images from S3")
        return images
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            raise Exception(f"S3 bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            raise Exception(f"Access denied to S3 bucket '{bucket_name}'")
        else:
            raise Exception(f"S3 error: {e}")
    except Exception as e:
        raise Exception(f"Failed to list S3 objects: {str(e)}")

def create_temp_config_file(config_data: Dict[str, Any], temp_dir: str) -> str:
    """
    Create a temporary config file from config data.
    
    Args:
        config_data: Configuration dictionary
        temp_dir: Temporary directory to save config file
    
    Returns:
        Path to the created config file
    """
    config_path = os.path.join(temp_dir, "inference_config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print_acc(f"Created temporary config file: {config_path}")
    return config_path

def upload_images_to_s3(image_paths: List[str], s3_output_dir: str) -> List[str]:
    """
    Upload image files to S3 and return the S3 URLs.
    
    Args:
        image_paths: List of local image file paths
        s3_output_dir: S3 directory path like "s3://bucket-name/path/to/folder"
    
    Returns:
        List of S3 URLs for uploaded images
    """
    if s3_client is None:
        raise Exception("S3 client not initialized. Check AWS credentials.")
    
    # Parse S3 output directory
    if not s3_output_dir.startswith('s3://'):
        raise ValueError("S3 output directory must start with 's3://'")
    
    # Remove s3:// prefix and split bucket/path
    s3_path = s3_output_dir[5:]  # Remove 's3://'
    if '/' in s3_path:
        bucket_name, folder_path = s3_path.split('/', 1)
        # Ensure folder path ends with /
        if folder_path and not folder_path.endswith('/'):
            folder_path += '/'
    else:
        bucket_name = s3_path
        folder_path = ''
    
    print_acc(f"Uploading {len(image_paths)} images to s3://{bucket_name}/{folder_path}")
    
    uploaded_urls = []
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            # Get the filename from the local path
            filename = os.path.basename(image_path)
            
            # Construct S3 key
            s3_key = folder_path + filename
            
            # Upload the file
            with open(image_path, 'rb') as img_file:
                s3_client.upload_fileobj(
                    img_file,
                    bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'image/png',  # Assuming PNG, adjust if needed
                        'ACL': 'private'  # Adjust ACL as needed
                    }
                )
            
            # Construct the S3 URL
            s3_url = f"s3://{bucket_name}/{s3_key}"
            uploaded_urls.append(s3_url)
            
            print_acc(f"Uploaded image {i}/{len(image_paths)}: {filename} -> {s3_url}")
            
        except Exception as e:
            print_acc(f"Error uploading image {image_path}: {e}")
            raise Exception(f"Failed to upload image {image_path}: {str(e)}")
    
    print_acc(f"Successfully uploaded {len(uploaded_urls)} images to S3")
    return uploaded_urls

def encode_images_to_base64(image_paths: List[str]) -> List[str]:
    """
    Encode image files to base64 strings for response.
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        List of base64 encoded image strings
    """
    encoded_images = []
    
    for image_path in image_paths:
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                encoded_images.append(img_base64)
        except Exception as e:
            print_acc(f"Error encoding image {image_path}: {e}")
            raise Exception(f"Failed to encode image {image_path}: {str(e)}")
    
    return encoded_images

def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    
    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")

def run_training_job(config_files, recover=False, name=None, log_file=None):
    """
    Run training jobs with given configuration files
    
    Args:
        config_files: List of config file paths
        recover: Continue running additional jobs even if a job fails
        name: Name to replace [name] tag in config file
        log_file: Log file to write output to
    
    Returns:
        dict: Results with jobs completed/failed counts and any errors
    """
    if log_file is not None:
        setup_log_to_file(log_file)
    
    if len(config_files) == 0:
        raise Exception("You must provide at least one config file")
    
    jobs_completed = 0
    jobs_failed = 0
    errors = []
    
    if accelerator.is_main_process:
        print_acc(f"Running {len(config_files)} job{'' if len(config_files) == 1 else 's'}")
    
    for config_file in config_files:
        try:
            job = get_job(config_file, name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            error_msg = f"Error running job {config_file}: {str(e)}"
            print_acc(error_msg)
            errors.append(error_msg)
            jobs_failed += 1
            
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                error_msg2 = f"Error running on_error: {str(e2)}"
                print_acc(error_msg2)
                errors.append(error_msg2)
            
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
                
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                error_msg = f"Error running on_error: {str(e2)}"
                print_acc(error_msg)
                errors.append(error_msg)
            
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
    
    print_end_message(jobs_completed, jobs_failed)
    
    return {
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "errors": errors,
        "success": jobs_failed == 0
    }

def run_inference_job(config_data, temp_dir):
    """
    Run inference job with FluxInference and LoRA support
    
    Args:
        config_data: Configuration dictionary
        temp_dir: Temporary directory for files
    
    Returns:
        dict: Results with generated images and metadata
    """
    if not FLUX_INFERENCE_AVAILABLE:
        raise Exception("FluxInference not available. Check flux_inference.py import.")
    
    try:
        print_acc("Starting Flux inference job with LoRA support...")
        
        # Create LoRA cache directory in temp_dir
        lora_cache_dir = os.path.join(temp_dir, "lora_cache")
        
        # Resolve all LoRA paths in the configuration
        print_acc("Resolving LoRA paths...")
        config_data = update_config_with_resolved_loras(config_data, lora_cache_dir)
        
        # Create temporary config file
        config_path = create_temp_config_file(config_data, temp_dir)
        
        # Extract configuration
        model_cfg = config_data.get("model", {})
        generate_cfg = config_data.get("generate", {})
        ip_adapter_cfg = config_data.get("ip_adapter", {})

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
            prompts = [""]
        
        # Handle IP-Adapter reference images if provided
        ip_adapter_image_paths = []
        if "reference_images" in ip_adapter_cfg:
            ref_images = ip_adapter_cfg["reference_images"]
            if ref_images:
                print_acc("Processing IP-Adapter reference images...")
                # For inference reference images, we typically don't remove background
                # but you can change this by setting remove_bg=True
                ip_adapter_image_paths = save_images_for_inference(ref_images, temp_dir, remove_bg=False)
                # Update config with local paths
                ip_adapter_cfg["reference_images"] = ip_adapter_image_paths
        
        # Get output directory from config
        output_dir_config = generate_cfg.get("output_dir", "outputs")
        is_s3_output = output_dir_config.startswith("s3://")
        
        if is_s3_output:
            # If S3 output, use local temp directory for generation, then upload
            local_output_dir = os.path.join(temp_dir, "outputs")
            s3_output_dir = output_dir_config
            print_acc(f"S3 output mode: Will upload to {s3_output_dir}")
        else:
            # Local output directory
            local_output_dir = os.path.join(temp_dir, "outputs")
            s3_output_dir = None
            print_acc(f"Local output mode: {local_output_dir}")
        
        # Log resolved LoRA information
        lora_summary = {
            "inference_loras": len(model_cfg.get("inference_lora_paths", [])),
            "face_loras": len(generate_cfg.get("face_lora_paths", [])),
            "upper_body_loras": len(generate_cfg.get("upper_body_lora_paths", []))
        }
        print_acc(f"LoRA Summary: {lora_summary}")
        
        # Initialize FluxInference with resolved LoRA paths
        flux = FluxInference(
            model_path=model_cfg.get("name_or_path", "black-forest-labs/FLUX.1-dev"),
            output_dir=local_output_dir,
            device=model_cfg.get("device", "cuda"),
            dtype=model_cfg.get("dtype", "float16"),
            face_inpainting=generate_cfg.get("face_inpainting", False),
            upper_body_inpainting=generate_cfg.get("upper_body_inpainting", False),
            face_lora_paths=generate_cfg.get("face_lora_paths", []),
            face_lora_strengths=generate_cfg.get("face_lora_strengths", []),
            upper_body_lora_paths=generate_cfg.get("upper_body_lora_paths", []),
            upper_body_lora_strengths=generate_cfg.get("upper_body_lora_strengths", []),
            ip_adapter_model_path=ip_adapter_cfg.get("model_path", "XLabs-AI/flux-ip-adapter"),
            ip_adapter_images=ip_adapter_image_paths,
            ip_adapter_scales=ip_adapter_cfg.get("scales", []),
            ip_adapter_target_size=tuple(ip_adapter_cfg.get("target_size", [512, 512])),
            low_vram=model_cfg.get("low_vram", False),
            config=config_data,
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
        
        # Find generated image files
        output_files = []
        if os.path.exists(local_output_dir):
            for file in os.listdir(local_output_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_files.append(os.path.join(local_output_dir, file))
        
        print_acc(f"Found {len(output_files)} generated images")
        
        # Handle output based on S3 configuration
        if is_s3_output and s3_output_dir:
            # Upload images to S3
            if not s3_init_success:
                raise Exception("S3 output requested but S3 client not initialized")
            
            try:
                s3_urls = upload_images_to_s3(output_files, s3_output_dir)
                print_acc(f"Successfully uploaded {len(s3_urls)} images to S3")
                
                return {
                    "success": True,
                    "images_generated": len(generated_images),
                    "output_type": "s3",
                    "output_location": s3_output_dir,
                    "s3_urls": s3_urls,
                    "filenames": [os.path.basename(f) for f in output_files],
                    "prompts_processed": len(prompts),
                    "config_used": config_data,
                    "ip_adapter_enabled": flux.ip_adapter_enabled,
                    "reference_images_used": len(ip_adapter_image_paths),
                    "lora_summary": lora_summary
                }
                
            except Exception as e:
                # Fallback to base64 if S3 upload fails
                print_acc(f"S3 upload failed, falling back to base64: {e}")
                encoded_images = encode_images_to_base64(output_files)
                
                return {
                    "success": True,
                    "images_generated": len(generated_images),
                    "output_type": "base64_fallback",
                    "s3_upload_error": str(e),
                    "images": encoded_images,
                    "filenames": [os.path.basename(f) for f in output_files],
                    "prompts_processed": len(prompts),
                    "config_used": config_data,
                    "ip_adapter_enabled": flux.ip_adapter_enabled,
                    "reference_images_used": len(ip_adapter_image_paths),
                    "lora_summary": lora_summary
                }
        else:
            # Return images as base64 (original behavior)
            encoded_images = encode_images_to_base64(output_files)
            
            return {
                "success": True,
                "images_generated": len(generated_images),
                "output_type": "base64",
                "images": encoded_images,
                "filenames": [os.path.basename(f) for f in output_files],
                "prompts_processed": len(prompts),
                "config_used": config_data,
                "ip_adapter_enabled": flux.ip_adapter_enabled,
                "reference_images_used": len(ip_adapter_image_paths),
                "lora_summary": lora_summary
            }
        
    except Exception as e:
        error_msg = f"Inference job failed: {str(e)}"
        print_acc(error_msg)
        raise Exception(error_msg)

# Initialize accelerator and load models/setup here before serverless starts
print_acc("Initializing accelerator and loading models...")
accelerator = get_accelerator()

# Initialize S3 client
s3_init_success = initialize_s3_client()

# Initialize rembg for background removal
rembg_init_success = initialize_rembg()

# Pre-load any models, tokenizers, or heavy resources that will be reused
# This happens once when the container starts, not on each request
def initialize_models():
    """Initialize and cache models/resources that will be reused across jobs"""
    try:
        print_acc("Pre-loading Flux model components...")
        return True
        
    except Exception as e:
        print_acc(f"Error during model initialization: {e}")
        print_acc("Training will continue but may need to load models during job execution")
        return False

# Initialize models when container starts
model_init_success = initialize_models()

def handler(job):
    """
    RunPod serverless handler function with support for both training and inference
    
    Expected input format:
    
    TRAINING (type: "train" or omitted for backward compatibility):
    {
        "type": "train",  # Optional, defaults to "train" 
        "images": [images_data] or {"s3_folder": ...} or "s3://bucket/folder/",
        "config_files": ["config/train_flux.yaml"],
        "recover": false,
        "name": "my_model_v1",
        "log_file": "training.log",
        "remove_background": true  # Optional, defaults to true for training
    }
    
    INFERENCE (type: "inference"):
    {
        "type": "inference",
        "config": {
            "model": {
                "name_or_path": "black-forest-labs/FLUX.1-dev",
                "device": "cuda",
                "dtype": "float16",
                "inference_lora_paths": [
                    "anuraj-sisyphus/avatar-loras/slaymnsha_face_v1.safetensors",
                    "/local/path/to/lora.safetensors"
                ],
                "inference_lora_strengths": [0.8, 0.6]
            },
            "generate": {
                "prompts": ["a beautiful woman in a red dress"],
                "negative_prompt": "ugly, bad anatomy",
                "width": 1024,
                "height": 1024,
                "sample_steps": 20,
                "guidance_scale": 7.0,
                "num_images": 1,
                "seed": -1,
                "face_lora_paths": [
                    "anuraj-sisyphus/avatar-loras/face_model.safetensors"
                ],
                "face_lora_strengths": [0.7],
                "upper_body_lora_paths": [
                    "user/repo/body_lora.safetensors"
                ],
                "upper_body_lora_strengths": [0.5]
            },
            "ip_adapter": {
                "model_path": "XLabs-AI/flux-ip-adapter",
                "reference_images": [images_data] or {"s3_folder": ...} or "s3://bucket/folder/",
                "scales": [0.6],
                "target_size": [512, 512]
            }
        }
    }
    """
    try:
        # Check if model initialization was successful
        if not model_init_success:
            return {
                "error": "Model initialization failed during container startup",
                "success": False
            }
        
        # Get input data
        job_input = job["input"]
        
        # Determine job type (default to "train" for backward compatibility)
        job_type = job_input.get("type", "train").lower()
        
        print_acc(f"Processing job type: {job_type}")
        
        # Route to appropriate handler based on type
        if job_type == "train":
            return handle_training_job(job_input)
        elif job_type == "inference":
            return handle_inference_job(job_input)
        else:
            return {
                "error": f"Unknown job type: {job_type}. Supported types: 'train', 'inference'",
                "success": False
            }
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print_acc(error_msg)
        return {
            "error": error_msg,
            "success": False,
            "job_type": job_input.get("type", "unknown")
        }

def handle_training_job(job_input):
    """
    Handle training job (original functionality with background removal)
    
    Args:
        job_input: Job input dictionary
    
    Returns:
        dict: Training job results
    """
    try:
        # Extract required parameters
        images_input = job_input.get("images")
        
        if not images_input:
            return {
                "error": "No images provided. 'images' field is required for training.",
                "success": False,
                "job_type": "train"
            }
        
        # Extract optional parameters
        config_files = job_input.get("config_files", ["config/train_flux.yaml"])
        recover = job_input.get("recover", False)
        name_tag = job_input.get("name", None)
        log_file = job_input.get("log_file", None)
        remove_bg = job_input.get("remove_background", False)  # Default to True for training
        
        # Log the name tag being used
        if name_tag:
            print_acc(f"Using name tag for replacement: '{name_tag}'")
        else:
            print_acc("No name tag provided - [name] placeholders will not be replaced")
        
        # Log background removal setting
        if remove_bg:
            if REMBG_AVAILABLE:
                print_acc("Background removal enabled")
            else:
                print_acc("Background removal requested but rembg not available")
        else:
            print_acc("Background removal disabled")
        
        # Validate config_files is a list
        if isinstance(config_files, str):
            config_files = [config_files]
        
        if not isinstance(config_files, list):
            return {
                "error": "config_files must be a string or list of strings",
                "success": False,
                "job_type": "train"
            }
        
        # Determine input type and log appropriately
        if isinstance(images_input, dict) and 's3_folder' in images_input:
            s3_folder = images_input['s3_folder']
            print_acc(f"Processing S3 folder: s3://{s3_folder.get('bucket')}/{s3_folder.get('folder', '')}")
        elif isinstance(images_input, str) and images_input.startswith('s3://') and images_input.endswith('/'):
            print_acc(f"Processing S3 folder: {images_input}")
        elif isinstance(images_input, list):
            print_acc(f"Processing {len(images_input)} individual images")
        else:
            return {
                "error": "Invalid images format. Use S3 folder object, S3 folder URL, or list of images.",
                "success": False,
                "job_type": "train"
            }
        
        if s3_init_success:
            print_acc("S3 support enabled")
        else:
            print_acc("S3 support disabled - set AWS credentials to enable")
        
        # Save images to datasets/input folder with background removal
        try:
            dataset_path = save_images_to_dataset(images_input, remove_bg=remove_bg)
        except Exception as e:
            return {
                "error": f"Failed to save images: {str(e)}",
                "success": False,
                "job_type": "train"
            }
        
        print_acc(f"Starting training with config files: {config_files}")
        print_acc(f"Images saved to: {dataset_path}")
        print_acc("Using pre-loaded models and accelerator...")
        
        # Run the training job
        result = run_training_job(
            config_files=config_files,
            recover=recover,
            name=name_tag,
            log_file=log_file
        )
        
        # Add dataset info to result
        result["dataset_path"] = dataset_path
        result["job_type"] = "train"
        # Count images in the dataset folder
        file_count = len([f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))])
        result["files_processed"] = file_count
        result["s3_enabled"] = s3_init_success
        result["rembg_enabled"] = rembg_init_success
        result["background_removal_applied"] = remove_bg and rembg_init_success
        result["name_tag_used"] = name_tag
        
        return result
        
    except Exception as e:
        error_msg = f"Training job error: {str(e)}"
        print_acc(error_msg)
        return {
            "error": error_msg,
            "success": False,
            "jobs_completed": 0,
            "jobs_failed": 1,
            "job_type": "train"
        }

def handle_inference_job(job_input):
    """
    Handle inference job using FluxInference with LoRA support
    
    Args:
        job_input: Job input dictionary
    
    Returns:
        dict: Inference job results
    """
    if not FLUX_INFERENCE_AVAILABLE:
        return {
            "error": "FluxInference not available. Ensure flux_inference.py is in the correct location.",
            "success": False,
            "job_type": "inference"
        }
    
    temp_dir = None
    try:
        # Extract config
        config_data = job_input.get("config")
        if not config_data:
            return {
                "error": "No config provided. 'config' field is required for inference.",
                "success": False,
                "job_type": "inference"
            }
        
        # Create temporary directory for this job
        temp_dir = tempfile.mkdtemp(prefix="flux_inference_")
        print_acc(f"Created temporary directory: {temp_dir}")

        # Handle S3 start_image download
        generate_cfg = config_data.get("generate", {})
        if generate_cfg.get("start_image"):
            start_image = generate_cfg["start_image"]
            if isinstance(start_image, str) and start_image.startswith('s3://'):
                if not s3_init_success:
                    return {
                        "error": "S3 credentials required for start_image but S3 client not initialized",
                        "success": False,
                        "job_type": "inference"
                    }
        
                # Download start_image from S3 and update config
                print_acc("Processing S3 start_image...")
                config_data = handle_start_image_s3(config_data, temp_dir)
        
        # Validate required config sections
        if "generate" not in config_data:
            return {
                "error": "Config must contain 'generate' section",
                "success": False,
                "job_type": "inference"
            }
        
        generate_cfg = config_data["generate"]
        if not generate_cfg.get("prompts"):
            return {
                "error": "No prompts provided in generate config",
                "success": False,
                "job_type": "inference"
            }
        
        print_acc(f"Starting inference with {len(generate_cfg['prompts'])} prompts")
        
        # Check for S3 support if needed
        ip_adapter_cfg = config_data.get("ip_adapter", {})
        ref_images = ip_adapter_cfg.get("reference_images")
        
        if ref_images:
            # Check if reference images require S3
            requires_s3 = False
            if isinstance(ref_images, dict) and 's3_folder' in ref_images:
                requires_s3 = True
            elif isinstance(ref_images, str) and ref_images.startswith('s3://'):
                requires_s3 = True
            elif isinstance(ref_images, list):
                for img in ref_images:
                    if isinstance(img, str) and img.startswith('s3://'):
                        requires_s3 = True
                        break
                    elif isinstance(img, dict) and ('bucket' in img and 'key' in img):
                        requires_s3 = True
                        break
            
            if requires_s3 and not s3_init_success:
                return {
                    "error": "S3 credentials required for reference images but S3 client not initialized",
                    "success": False,
                    "job_type": "inference"
                }
        
        # Run inference
        result = run_inference_job(config_data, temp_dir)
        result["job_type"] = "inference"
        result["s3_enabled"] = s3_init_success
        result["rembg_enabled"] = rembg_init_success
        
        return result
        
    except Exception as e:
        error_msg = f"Inference job error: {str(e)}"
        print_acc(error_msg)
        return {
            "error": error_msg,
            "success": False,
            "job_type": "inference"
        }
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print_acc(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print_acc(f"Warning: Could not clean up temp directory {temp_dir}: {cleanup_error}")

# Start the serverless function
runpod.serverless.start({"handler": handler})