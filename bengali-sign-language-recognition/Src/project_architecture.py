"""
BaSL Glove — Complete Pipeline Integration
Demonstrates how to use input_collect and input_process together

Usage:
    python project_architecture.py --port COM6
"""

import numpy as np
import argparse
import os

# Import modules
from input_collect import DataCollector
from input_process import InputProcessor

def main():
    parser = argparse.ArgumentParser(description='BaSL Glove Complete Pipeline')
    parser.add_argument('--port', required=True, help='Serial port')
    parser.add_argument('--baud', default=115200, type=int, help='Baud rate')
    parser.add_argument('--samples', default=1, type=int, help='Number of samples')
    parser.add_argument('--model', default='models/xgboost_model.pkl')
    parser.add_argument('--scaler', default='models/scaler.pkl')
    parser.add_argument('--encoder', default='models/label_encoder.pkl')
    parser.add_argument('--save-raw', help='Save raw data to .npy file')
    args = parser.parse_args()
    
    print("="*50)
    print("🧤 BaSL Glove - Complete Pipeline")
    print("="*50)
    
    # Step 1: Collect data from glove
    print("\n📊 STEP 1: Collecting data from glove...")
    collector = DataCollector(args.port, args.baud)
    
    if not collector.connect():
        print("❌ Failed to connect to glove")
        return
    
    samples = collector.collect_multiple(num_samples=args.samples)
    collector.disconnect()
    
    if not samples:
        print("❌ No samples collected")
        return
    
    # Convert to numpy array
    raw_data = np.array(samples)
    print(f"\n✅ Collected {len(raw_data)} samples")
    print(f"   Shape: {raw_data.shape}")
    
    # Save raw data if requested
    if args.save_raw:
        np.save(args.save_raw, raw_data)
        print(f"💾 Raw data saved to: {args.save_raw}")
    
    # Step 2: Process and predict
    print("\n📊 STEP 2: Processing and predicting...")
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    processor = InputProcessor(args.model, args.scaler, args.encoder)
    results = processor.process_from_raw(raw_data)
    
    # Step 3: Display summary
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    print(f"  Samples collected: {len(samples)}")
    print(f"  Predictions made: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"\n  Sample {i+1}:")
        print(f"    Predicted: {result['predicted_label']}")
        print(f"    Confidence: {result['confidence']:.2f}%")

if __name__ == "__main__":
    main()