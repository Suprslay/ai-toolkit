import cv2
import mediapipe as mp
import numpy as np
import os

class BodyCropper:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model (better for distant faces)
            min_detection_confidence=0.2  # Lower threshold for better detection
        )
        print("[Face Detection] Initialized with full range model, confidence=0.2")
    
    def detect_faces_robust(self, img):
        """
        Enhanced face detection using the same logic as your inpainting script.
        """
        img_rgb = np.array(img)
        height, width, _ = img_rgb.shape
        
        print(f"[Face Detection] Image size: {width}x{height}")
        
        results = self.face_detector.process(img_rgb)
        
        boxes = []
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
                
                # Face size filtering
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
                
                # Apply same expansion as inpainting script
                exp = 0.05
                nx = max(0, int(x - w * exp))
                ny = max(0, int(y - h * exp))
                nw = min(width - nx, int(w * (1 + 2 * exp)))
                nh = min(height - ny, int(h * (1 + 2 * exp)))
                
                print(f"      ↳ Final bbox: x={nx}, y={ny}, w={nw}, h={nh}")
                boxes.append((nx, ny, nw, nh))
        else:
            print("[Face Detection] No detections found by MediaPipe")
        
        # Sort by size (largest first)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        
        print(f"[Face Detection] Final result: {len(boxes)} faces passed filtering")
        return boxes
    
    def crop_below_face(self, image_path, output_path=None, margin_factor=0.005, debug=False):
        """
        Simple crop: detect face bounding box and keep everything below it.
        
        Args:
            image_path: Path to input image
            output_path: Path to save cropped image (optional)
            margin_factor: Additional margin below face box (0.02 = 2% buffer space)
            debug: Save debug images showing detected face
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB for MediaPipe
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Detect faces
        face_boxes = self.detect_faces_robust(pil_image)
        
        if not face_boxes:
            print("[Crop] No faces detected - cannot determine crop boundary")
            return None
        
        # Use the largest face
        largest_face = max(face_boxes, key=lambda b: b[2] * b[3])
        face_x, face_y, face_w, face_h = largest_face
        
        # Crop from bottom of face box
        face_bottom = face_y + face_h
        
        # Add small margin
        margin = 10
        crop_start = min(height, face_bottom + margin)
        
        print(f"[Crop] Face bottom at y={face_bottom}")
        print(f"[Crop] Cropping from y={crop_start} (+ {margin}px margin)")
        
        # Save debug image if requested
        if debug and output_path:
            debug_image = image.copy()
            
            # Draw face box
            cv2.rectangle(debug_image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
            cv2.putText(debug_image, "Face", (face_x, face_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw face bottom line
            cv2.line(debug_image, (0, face_bottom), (width, face_bottom), (255, 165, 0), 2)
            cv2.putText(debug_image, "Face Bottom", (10, face_bottom-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Draw crop line  
            cv2.line(debug_image, (0, crop_start), (width, crop_start), (0, 0, 255), 3)
            cv2.putText(debug_image, "Crop Line", (10, crop_start+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            debug_dir = os.path.dirname(output_path) if output_path else "."
            debug_filename = f"debug_face_crop_{os.path.basename(image_path)}"
            debug_path = os.path.join(debug_dir, debug_filename)
            cv2.imwrite(debug_path, debug_image)
            print(f"[Debug] Saved debug image: {debug_path}")
        
        # Crop everything below the face
        cropped_image = image[crop_start:, :]
        
        if cropped_image.size == 0:
            print("[Crop] Warning: Cropped image is empty - face too low in image")
            return None
        
        print(f"[Crop] Final cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        
        # Save or return the cropped image
        if output_path:
            cv2.imwrite(output_path, cropped_image)
            print(f"Cropped image saved to: {output_path}")
        
        return cropped_image
    
    def batch_crop(self, input_folder, output_folder, margin_factor=0.02, debug=False):
        """
        Crop multiple images in a folder, keeping everything below face.
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save cropped images
            margin_factor: Additional margin below face
            debug: Save debug images showing detection results
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        success_count = 0
        total_count = 0
        
        # Process each image in the input folder
        for filename in os.listdir(input_folder):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_extensions:
                total_count += 1
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"body_below_face_{filename}")
                
                try:
                    result = self.crop_below_face(input_path, output_path, margin_factor, debug)
                    if result is not None:
                        success_count += 1
                        print(f"Successfully processed: {filename}")
                    else:
                        print(f"Failed to process: {filename} (no face detected)")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        print(f"\n[Batch] Completed: {success_count}/{total_count} images processed successfully")

def main():
    # Example usage
    cropper = BodyCropper()
    
    # Single image example - simple face-based cropping
    try:
        # Replace with your image path
        input_image = "outputs/001_professional_fashion_photography_42_0.png"
        output_image = "body_below_face.jpg"
        
        print("Simple face detection and body cropping...")
        print("• Detects face bounding box")
        print("• Crops everything below the face box")
        print("• Keeps neck, shoulders, and entire body")
        
        cropped = cropper.crop_below_face(
            input_image, 
            output_image, 
            margin_factor=0.02,  # 2% margin below face
            debug=True
        )
        
        if cropped is not None:
            print("Body portion extracted successfully!")
            print("\nAdjustment options:")
            print("  • Increase margin_factor (0.03-0.05) for more space below face")
            print("  • Decrease margin_factor (0.001-0.01) for tighter crop")
            
            # Display the result (optional)
            original = cv2.imread(input_image)
            if original is not None:
                cv2.imshow("Original", original)
                cv2.imshow("Body Below Face", cropped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Failed to extract body portion - no face detected")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure image contains a visible face")
        print("2. Check image quality and lighting")
        print("3. Install MediaPipe: pip install mediapipe")
    
    # Batch processing example (uncomment to use)
    # print("\nRunning batch processing...")
    # cropper.batch_crop("input_images/", "output_images/", debug=True)

if __name__ == "__main__":
    main()