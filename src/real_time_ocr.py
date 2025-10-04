import cv2
import paddle
import numpy as np
from paddleocr import PaddleOCR
import yaml
import time
from typing import Dict, List, Optional
import logging

class CasinoOCRProcessor:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.load_config(config_path)
        self.setup_logging()
        self.init_ocr_model()
        self.setup_database()
        
    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def init_ocr_model(self):
        """Initialize PaddlePaddle OCR model with GPU support"""
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.config['model']['use_gpu'],
                gpu_mem=self.config['model']['gpu_mem'],
                det_model_dir=self.config['model']['det_model_dir'],
                rec_model_dir=self.config['model']['rec_model_dir'],
                cls_model_dir=self.config['model']['cls_model_dir']
            )
            self.logger.info("PaddlePaddle OCR model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR model: {e}")
            raise
    
    def setup_database(self):
        """Initialize database connection"""
        # This would be implemented based on your specific database
        self.db_handler = DatabaseHandler(self.config['database'])
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised
    
    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """Extract Region of Interest where prize amounts are displayed"""
        roi_coords = self.config['processing']['roi_coordinates']
        x1, y1, x2, y2 = roi_coords
        return image[y1:y2, x1:x2]
    
    def parse_prize_amount(self, text: str) -> Optional[float]:
        """Parse and validate prize amount from OCR text"""
        import re
        
        # Remove common noise characters
        clean_text = re.sub(r'[^\d.,$€£]', '', text)
        
        # Extract numeric values with currency symbols
        patterns = [
            r'[\$€£]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)[\$€£]?',  # 1,234.56$
            r'[\$€£]?(\d+(?:\.\d{2})?)',  # $1234.56
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, clean_text)
            if matches:
                try:
                    # Remove commas and convert to float
                    amount_str = matches[0].replace(',', '')
                    amount = float(amount_str)
                    
                    # Validate reasonable prize range
                    if 0.01 <= amount <= 1000000:  # $0.01 to $1,000,000
                        return amount
                except ValueError:
                    continue
        
        return None
    
    def process_frame(self, frame: np.ndarray, machine_id: str) -> Dict:
        """Process a single frame and extract prize information"""
        try:
            # Extract ROI
            roi_image = self.extract_roi(frame)
            
            # Preprocess
            processed_image = self.preprocess_image(roi_image)
            
            # Perform OCR
            result = self.ocr.ocr(processed_image, cls=True)
            
            extracted_data = {
                'machine_id': machine_id,
                'timestamp': time.time(),
                'prize_amount': None,
                'confidence': 0.0,
                'raw_text': [],
                'status': 'processing'
            }
            
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    extracted_data['raw_text'].append({
                        'text': text,
                        'confidence': confidence
                    })
                    
                    # Try to parse prize amount
                    if confidence >= self.config['processing']['confidence_threshold']:
                        prize_amount = self.parse_prize_amount(text)
                        if prize_amount and (extracted_data['prize_amount'] is None or 
                                           confidence > extracted_data['confidence']):
                            extracted_data['prize_amount'] = prize_amount
                            extracted_data['confidence'] = confidence
            
            # Update status
            if extracted_data['prize_amount'] is not None:
                extracted_data['status'] = 'success'
            else:
                extracted_data['status'] = 'no_prize_found'
                
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return {
                'machine_id': machine_id,
                'timestamp': time.time(),
                'prize_amount': None,
                'confidence': 0.0,
                'raw_text': [],
                'status': f'error: {str(e)}'
            }
    
    def update_database(self, result: Dict):
        """Update database with extracted prize information"""
        if result['status'] == 'success' and result['prize_amount'] is not None:
            try:
                self.db_handler.update_prize(
                    machine_id=result['machine_id'],
                    prize_amount=result['prize_amount'],
                    confidence=result['confidence'],
                    timestamp=result['timestamp']
                )
                self.logger.info(f"Updated database: Machine {result['machine_id']} - ${result['prize_amount']}")
            except Exception as e:
                self.logger.error(f"Database update failed: {e}")