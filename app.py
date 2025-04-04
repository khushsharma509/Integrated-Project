from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import re
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import logging

app = Flask(__name__)
CORS(app)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODELS_LOADED = False
sanskrit_model = None
digit_model = None

try:
    sanskrit_model = load_model("resnext_devanagari_classifier.h5")
    logger.info("✅ Sanskrit model loaded successfully")
    logger.info(f"Sanskrit model input shape: {sanskrit_model.input_shape}")
    
    digit_model = load_model("resnext_digit_classifier.h5")
    logger.info("✅ Digit model loaded successfully")
    logger.info(f"Digit model input shape: {digit_model.input_shape}")
    
    MODELS_LOADED = True
except Exception as e:
    logger.error(f"❌ Error loading models: {str(e)}")

# Character mappings
SANSKRIT_MAPPING = {
    1: {'character': 'क', 'latin': 'ka', 'name': 'ka'},
    2: {'character': 'ख', 'latin': 'kha', 'name': 'kha'},
    3: {'character': 'ग', 'latin': 'ga', 'name': 'ga'},
    4: {'character': 'घ', 'latin': 'gha', 'name': 'gha'},
    5: {'character': 'ङ', 'latin': 'kna', 'name': 'kna'},
    6: {'character': 'च', 'latin': 'cha', 'name': 'cha'},
    7: {'character': 'छ', 'latin': 'chha', 'name': 'chha'},
    8: {'character': 'ज', 'latin': 'ja', 'name': 'ja'},
    9: {'character': 'झ', 'latin': 'jha', 'name': 'jha'},
    0: {'character': 'ञ', 'latin': 'yna', 'name': 'yna'}
}

DIGIT_MAPPING = {
    0: {'character': '०', 'latin': '0', 'name': 'Śūnya'},
    1: {'character': '१', 'latin': '1', 'name': 'ēka'},
    2: {'character': '२', 'latin': '2', 'name': 'duī'},
    3: {'character': '३', 'latin': '3', 'name': 'tīna'},
    4: {'character': '४', 'latin': '4', 'name': 'cāra'},
    5: {'character': '५', 'latin': '5', 'name': 'pām̐ca'},
    6: {'character': '६', 'latin': '6', 'name': 'cha'},
    7: {'character': '७', 'latin': '7', 'name': 'sāta'},
    8: {'character': '८', 'latin': '8', 'name': 'āṭha'},
    9: {'character': '९', 'latin': '9', 'name': 'nau'}
}

def preprocess_sanskrit_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Failed to read Sanskrit image")
            return None
        
        logger.info(f"Original Sanskrit image shape: {img.shape}")
        cv2.imwrite('debug_sanskrit_original.png', img)
        
        # Resize and normalize
        img = cv2.resize(img, (32, 32))
        img = img.astype('float32') / 255.0
        
        
        if np.mean(img) > 0.5:
            img = 1 - img
        
        cv2.imwrite('debug_sanskrit_processed.png', (img * 255).astype(np.uint8))
        return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
    except Exception as e:
        logger.error(f"Sanskrit preprocessing error: {str(e)}")
        return None
 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-sanskrit', methods=['POST'])
def predict_sanskrit():
    if not MODELS_LOADED or sanskrit_model is None:
        return jsonify({'success': False, 'message': 'Sanskrit model not loaded'})
    
    temp_path = 'temp_sanskrit.png'
    
    try:
        # Validate input
        if not request.json or 'image' not in request.json:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Extract and save image
        img_data = request.json['image']
        match = re.search(r'base64,(.*)', img_data)
        if not match:
            return jsonify({'success': False, 'message': 'Invalid image format'})
        
        img_str = match.group(1)
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
            if not os.path.exists(temp_path):
                raise ValueError("Failed to save temporary file")
        except Exception as e:
            return jsonify({'success': False, 'message': f'Failed to save image: {str(e)}'})
        
        
        processed_img = preprocess_sanskrit_image(temp_path)
        if processed_img is None:
            return jsonify({'success': False, 'message': 'Image processing failed'})
        
        try:
            prediction = sanskrit_model.predict(processed_img)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            logger.info(f"Sanskrit prediction scores: {prediction}")
            logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
            
            result = SANSKRIT_MAPPING.get(predicted_class, {'character': '?', 'latin': '?', 'name': 'Unknown'})
            return jsonify({
                'success': True,
                'prediction': {
                    **result,
                    'confidence': round(confidence * 100, 2)
                }
            })
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error during prediction'
            })
            
    except Exception as e:
        logger.error(f"Unexpected error in Sanskrit prediction: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Unexpected error processing Sanskrit character'
        })
    finally:
        
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/predict-digit', methods=['POST'])
def predict_digit():
    if not MODELS_LOADED or digit_model is None:
        return jsonify({'success': False, 'message': 'Digit model not loaded'})
    
    temp_path = 'temp_digit.png'
    
    try:
        
        if not request.json or 'image' not in request.json:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        
        img_data = request.json['image']
        match = re.search(r'base64,(.*)', img_data)
        if not match:
            return jsonify({'success': False, 'message': 'Invalid image format'})
        
        img_str = match.group(1)
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
            if not os.path.exists(temp_path):
                raise ValueError("Failed to save temporary file")
        except Exception as e:
            return jsonify({'success': False, 'message': f'Failed to save image: {str(e)}'})
        
        
        processed_img = preprocess_sanskrit_image(temp_path)
        if processed_img is None:
            return jsonify({'success': False, 'message': 'Image processing failed'})
        
        
        try:
            prediction = digit_model.predict(processed_img)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            logger.info(f"Digit prediction scores: {prediction}")
            logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
            
            if confidence < 0.1:  # Very low confidence threshold
                return jsonify({
                    'success': False,
                    'message': 'Unable to recognize digit. Please try drawing more clearly'
                })
            
            result = DIGIT_MAPPING.get(predicted_class, {
                'character': '?',
                'latin': '?',
                'name': 'Unknown'
            })
            
            return jsonify({
                'success': True,
                'prediction': {
                    **result,
                    'confidence': round(confidence * 100, 2)
                }
            })
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error during prediction'
            })
            
    except Exception as e:
        logger.error(f"Unexpected error in digit prediction: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Unexpected error processing digit'
        })
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)