"""RunPod Serverless Handler for Flux Training"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
import runpod
import base64
import requests
from PIL import Image
import io
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

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

# Initialize S3 client globally
s3_client = None

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

def list_s3_images(bucket_name, folder_path=""):
    """
    List all image files in an S3 bucket folder
    
    Args:
        bucket_name: S3 bucket name
        folder_path: Folder path within bucket (optional)
    
    Returns:
        List of S3 object keys for image files
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
        
        print_acc(f"Found {len(image_keys)} image files in s3://{bucket_name}/{folder_path}")
        return image_keys
        
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

def download_s3_folder_images(bucket_name, folder_path=""):
    """
    Download all images from an S3 folder
    
    Args:
        bucket_name: S3 bucket name
        folder_path: Folder path within bucket (optional)
    
    Returns:
        List of PIL Image objects
    """
    # Get list of image keys
    image_keys = list_s3_images(bucket_name, folder_path)
    
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

def save_images_to_dataset(images_input):
    """
    Save images to datasets/input folder as 1.png, 2.png, etc.
    
    Args:
        images_input: Can be:
            - List of image data (base64 strings, URLs, S3 URLs, or objects)
            - S3 folder object: {"s3_folder": {"bucket": "name", "folder": "path"}}
            - S3 folder URL: "s3://bucket-name/folder-path/"
    
    Returns:
        str: Path to the created dataset folder
    """
    # Create dataset directory
    dataset_path = "datasets/input"
    os.makedirs(dataset_path, exist_ok=True)
    
    images = []
    
    # Handle S3 folder input
    if isinstance(images_input, dict) and 's3_folder' in images_input:
        # S3 folder object format: {"s3_folder": {"bucket": "name", "folder": "path"}}
        s3_folder = images_input['s3_folder']
        bucket_name = s3_folder.get('bucket')
        folder_path = s3_folder.get('folder', '')
        
        if not bucket_name:
            raise Exception("S3 folder object must have 'bucket' field")
        
        print_acc(f"Downloading all images from S3 folder: s3://{bucket_name}/{folder_path}")
        images = download_s3_folder_images(bucket_name, folder_path)
        
    elif isinstance(images_input, str) and images_input.startswith('s3://') and images_input.endswith('/'):
        # S3 folder URL format: "s3://bucket-name/folder-path/"
        url_parts = images_input[5:].rstrip('/').split('/', 1)
        bucket_name = url_parts[0]
        folder_path = url_parts[1] if len(url_parts) > 1 else ''
        
        print_acc(f"Downloading all images from S3 folder: {images_input}")
        images = download_s3_folder_images(bucket_name, folder_path)
        
    elif isinstance(images_input, list):
        # Original behavior - list of individual images
        print_acc(f"Processing {len(images_input)} individual images")
        
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
                
                images.append(image)
                
            except Exception as e:
                error_msg = f"Error processing image {i}: {str(e)}"
                print_acc(error_msg)
                raise Exception(error_msg)
    else:
        raise Exception("Invalid images input format")
    
    # Save all images to disk
    print_acc(f"Saving {len(images)} images to {dataset_path}")
    
    for i, image in enumerate(images, 1):
        try:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as PNG with sequential naming
            image_path = os.path.join(dataset_path, f"{i}.png")
            image.save(image_path, "PNG")
            print_acc(f"Saved image {i} to {image_path}")
            
        except Exception as e:
            error_msg = f"Error saving image {i}: {str(e)}"
            print_acc(error_msg)
            raise Exception(error_msg)
    
    print_acc(f"Successfully saved {len(images)} images to {dataset_path}")
    return dataset_path

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

# Initialize accelerator and load models/setup here before serverless starts
print_acc("Initializing accelerator and loading models...")
accelerator = get_accelerator()

# Initialize S3 client
s3_init_success = initialize_s3_client()

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
    RunPod serverless handler function
    
    Expected input format:
    
    Option 1 - S3 Folder (Object format):
    {
        "images": {"s3_folder": {"bucket": "my-bucket", "folder": "training-images"}},
        "config_files": ["config/train_flux.yaml"]
    }
    
    Option 2 - S3 Folder (URL format):
    {
        "images": "s3://my-bucket/training-images/",
        "config_files": ["config/train_flux.yaml"]
    }
    
    Option 3 - Individual images (original behavior):
    {
        "images": [
            "base64_string_1",
            "https://url-to-image.jpg", 
            "s3://my-bucket/single-image.jpg",
            {"bucket": "my-bucket", "key": "single-image.jpg"}
        ],
        "config_files": ["config/train_flux.yaml"]
    }
    
    Optional parameters:
    {
        "recover": false,
        "name": "my_model_v1",  # Name tag to replace [name] placeholders in config files
        "log_file": "training.log"
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
        
        # Extract required parameters
        images_input = job_input.get("images")
        
        if not images_input:
            return {
                "error": "No images provided. 'images' field is required.",
                "success": False
            }
        
        # Extract optional parameters
        config_files = job_input.get("config_files", ["config/train_flux.yaml"])
        recover = job_input.get("recover", False)
        name_tag = job_input.get("name", None)  # Name tag to replace [name] in config
        log_file = job_input.get("log_file", None)
        
        # Log the name tag being used
        if name_tag:
            print_acc(f"Using name tag for replacement: '{name_tag}'")
        else:
            print_acc("No name tag provided - [name] placeholders will not be replaced")
        
        # Validate config_files is a list
        if isinstance(config_files, str):
            config_files = [config_files]
        
        if not isinstance(config_files, list):
            return {
                "error": "config_files must be a string or list of strings",
                "success": False
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
                "success": False
            }
        
        if s3_init_success:
            print_acc("S3 support enabled")
        else:
            print_acc("S3 support disabled - set AWS credentials to enable")
        
        # Save images to datasets/input folder
        try:
            dataset_path = save_images_to_dataset(images_input)
        except Exception as e:
            return {
                "error": f"Failed to save images: {str(e)}",
                "success": False
            }
        
        print_acc(f"Starting training with config files: {config_files}")
        print_acc(f"Images saved to: {dataset_path}")
        print_acc("Using pre-loaded models and accelerator...")
        
        # Run the training job
        result = run_training_job(
            config_files=config_files,
            recover=recover,
            name=name_tag,  # Pass the name tag for [name] replacement
            log_file=log_file
        )
        
        # Add dataset info to result
        result["dataset_path"] = dataset_path
        # Count images in the dataset folder
        image_count = len([f for f in os.listdir(dataset_path) if f.endswith('.png')])
        result["images_processed"] = image_count
        result["s3_enabled"] = s3_init_success
        result["name_tag_used"] = name_tag  # Include the name tag in response
        
        return result
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print_acc(error_msg)
        return {
            "error": error_msg,
            "success": False,
            "jobs_completed": 0,
            "jobs_failed": 1
        }

# Start the serverless function
runpod.serverless.start({"handler": handler})