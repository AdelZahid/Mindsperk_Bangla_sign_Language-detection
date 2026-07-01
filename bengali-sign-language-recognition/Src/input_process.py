"""
BaSL Glove — Input Processing & Prediction Module
Preprocesses raw sensor data and predicts using trained model

Usage:
    python input_process.py --input raw_data.npy --model models/xgboost_model.pkl --scaler scaler.pkl --encoder label_encoder.pkl

    # Or with serial collection integrated
    python input_process.py --port COM6 --model models/xgboost_model.pkl --scaler scaler.pkl --encoder label_encoder.pkl
"""

import numpy as np
import joblib
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import collector if available
try:
    from input_collect import DataCollector
    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False

# ============================================
# CONSTANTS
# ============================================

NUM_TIMESTEPS = 20
NUM_FEATURES = 11
TOTAL_FEATURES = NUM_TIMESTEPS * NUM_FEATURES  # 220

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

def load_raw_data(filepath):
    """
    Load raw data from .npy file
    Returns: numpy array of shape (N, 20, 11) or (20, 11)
    """
    data = np.load(filepath)
    # If single sample, add batch dimension
    if len(data.shape) == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])
    return data

def preprocess_sample(sample, scaler):
    """
    Preprocess a (20, 11) sample using the same pipeline as training:
    1. Flatten to (220,)
    2. Reshape to (1, 220)
    3. Apply StandardScaler
    Returns: (1, 220) scaled features
    """
    # Flatten: (20, 11) → (220,)
    sample_flat = sample.flatten()
    
    # Reshape to 2D: (1, 220)
    sample_2d = sample_flat.reshape(1, -1)
    
    # Apply StandardScaler (fitted on training data)
    sample_scaled = scaler.transform(sample_2d)
    
    return sample_scaled

def preprocess_batch(samples, scaler):
    """
    Preprocess multiple samples
    samples: numpy array of shape (N, 20, 11)
    Returns: numpy array of shape (N, 220) scaled
    """
    N = samples.shape[0]
    samples_flat = samples.reshape(N, -1)  # (N, 220)
    samples_scaled = scaler.transform(samples_flat)  # (N, 220)
    return samples_scaled

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_sample(sample, model, scaler, label_encoder):
    """
    Predict sign from a single (20, 11) sample
    Returns: dict with prediction results
    """
    # Preprocess
    sample_scaled = preprocess_sample(sample, scaler)
    
    # Predict
    prediction = model.predict(sample_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(sample_scaled)[0]
    confidence = np.max(probabilities) * 100
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_labels = label_encoder.inverse_transform(top_3_indices)
    top_3_confidences = probabilities[top_3_indices] * 100
    
    return {
        'predicted_label': predicted_label,
        'confidence': confidence,
        'top_3': list(zip(top_3_labels, top_3_confidences)),
        'probabilities': probabilities
    }

def predict_batch(samples, model, scaler, label_encoder):
    """
    Predict signs from multiple samples
    samples: numpy array of shape (N, 20, 11)
    Returns: list of prediction results
    """
    # Preprocess all samples
    samples_scaled = preprocess_batch(samples, scaler)
    
    # Predict all
    predictions = model.predict(samples_scaled)
    probabilities = model.predict_proba(samples_scaled)
    
    results = []
    for i, pred in enumerate(predictions):
        predicted_label = label_encoder.inverse_transform([pred])[0]
        confidence = np.max(probabilities[i]) * 100
        
        # Top 3
        top_3_indices = np.argsort(probabilities[i])[-3:][::-1]
        top_3_labels = label_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[i][top_3_indices] * 100
        
        results.append({
            'predicted_label': predicted_label,
            'confidence': confidence,
            'top_3': list(zip(top_3_labels, top_3_confidences)),
            'probabilities': probabilities[i]
        })
    
    return results

def display_result(result, index=0):
    """Display prediction result in a formatted way"""
    print(f"\n{'='*50}")
    print(f"🎯 PREDICTION {index + 1}")
    print(f"{'='*50}")
    print(f"  Predicted: {result['predicted_label']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"{'-'*50}")
    print(f"  🔝 Top 3 predictions:")
    for label, conf in result['top_3']:
        bar = '█' * int(conf / 5)
        print(f"    {label:12s}: {conf:6.2f}% [{bar:20s}]")
    print(f"{'='*50}\n")

# ============================================
# MAIN PROCESSOR CLASS
# ============================================

class InputProcessor:
    def __init__(self, model_path, scaler_path, encoder_path):
        """
        Initialize processor with trained model components
        """
        print("📂 Loading model components...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        print(f"  ✅ Model loaded: {len(self.label_encoder.classes_)} classes")
        print(f"  📋 Classes: {', '.join(self.label_encoder.classes_)}")
        print(f"  🔢 Features: {TOTAL_FEATURES} ({NUM_TIMESTEPS} timesteps × {NUM_FEATURES} features)")
    
    def process_from_file(self, input_file):
        """
        Process and predict from a .npy file
        """
        print(f"\n📂 Loading data from: {input_file}")
        samples = load_raw_data(input_file)
        print(f"  Loaded {samples.shape[0]} samples")
        
        results = predict_batch(samples, self.model, self.scaler, self.label_encoder)
        
        # Display results
        print(f"\n📊 PREDICTION RESULTS")
        print(f"{'='*50}")
        for i, result in enumerate(results):
            display_result(result, i)
        
        return results
    
    def process_from_serial(self, port, baud=115200, num_samples=1):
        """
        Collect data from serial and predict
        Requires input_collect module
        """
        if not COLLECTOR_AVAILABLE:
            print("❌ input_collect module not available. Please install/import it.")
            return None
        
        # Collect data
        collector = DataCollector(port, baud)
        if not collector.connect():
            return None
        
        try:
            samples = collector.collect_multiple(num_samples=num_samples)
            
            if not samples:
                print("❌ No samples collected")
                return None
            
            # Convert to numpy array
            samples_array = np.array(samples)
            print(f"\n  ✅ Collected {len(samples_array)} samples")
            print(f"  Sample shape: {samples_array.shape}")
            
            # Predict
            results = predict_batch(samples_array, self.model, self.scaler, self.label_encoder)
            
            # Display results
            print(f"\n📊 PREDICTION RESULTS")
            print(f"{'='*50}")
            for i, result in enumerate(results):
                display_result(result, i)
            
            return results
            
        finally:
            collector.disconnect()
    
    def process_from_raw(self, raw_data):
        """
        Process raw numpy array (for integration)
        raw_data: numpy array of shape (20, 11) or (N, 20, 11)
        """
        # Ensure batch format
        if len(raw_data.shape) == 2:
            raw_data = raw_data.reshape(1, raw_data.shape[0], raw_data.shape[1])
        
        results = predict_batch(raw_data, self.model, self.scaler, self.label_encoder)
        return results

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='BaSL Glove - Input Processor & Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from saved .npy file
  python input_process.py --input raw_data.npy --model models/xgboost_model.pkl --scaler scaler.pkl --encoder label_encoder.pkl
  
  # Collect and predict from serial
  python input_process.py --port COM6 --model models/xgboost_model.pkl --scaler scaler.pkl --encoder label_encoder.pkl
  
  # Collect multiple samples and predict
  python input_process.py --port COM6 --samples 3 --model models/xgboost_model.pkl --scaler scaler.pkl --encoder label_encoder.pkl
        """
    )
    
    parser.add_argument('--input', help='Input .npy file with raw data')
    parser.add_argument('--port', help='Serial port for real-time collection')
    parser.add_argument('--baud', default=115200, type=int, help='Baud rate')
    parser.add_argument('--samples', default=1, type=int, help='Number of samples to collect')
    parser.add_argument('--model', required=True, help='Path to XGBoost model (.pkl)')
    parser.add_argument('--scaler', required=True, help='Path to StandardScaler (.pkl)')
    parser.add_argument('--encoder', required=True, help='Path to label encoder (.pkl)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("🧤 BaSL Glove - Input Processor")
    print("="*50)
    
    # Initialize processor
    processor = InputProcessor(args.model, args.scaler, args.encoder)
    
    # Mode 1: Process from file
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ File not found: {args.input}")
            return
        processor.process_from_file(args.input)
    
    # Mode 2: Process from serial
    elif args.port:
        if not COLLECTOR_AVAILABLE:
            print("❌ input_collect module not available. Please ensure input_collect.py is in the same directory.")
            print("   Or use --input to process existing files.")
            return
        processor.process_from_serial(args.port, args.baud, args.samples)
    
    else:
        print("❌ Please specify either --input or --port")
        print("   Use --help for more information.")

if __name__ == "__main__":
    main()