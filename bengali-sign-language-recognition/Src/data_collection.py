"""
BaSL Glove — Data Collection Script
Collects sensor data from glove and saves as CSV format with 20 timesteps per sample

Usage:
    python data_collection.py --port COM6 --label salam --output dataset.csv

Install dependency:
    pip install pyserial
"""

import serial
import argparse
import csv
import os
import time
from datetime import datetime

# ============================================
# CONSTANTS
# ============================================

HEADER = [
    "sample_id", "label", "timestep",
    "flex1", "flex2", "flex3", "flex4", "flex5",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]

NUM_TIMESTEPS = 20
SAMPLE_INTERVAL_MS = 100  # 100ms between readings (matches training)

# ============================================
# DATA COLLECTION FUNCTIONS
# ============================================

def get_next_sample_id(csv_file):
    """Get the next sample_id from existing CSV file"""
    if not os.path.isfile(csv_file):
        return 1
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            sample_ids = []
            for row in reader:
                if row:
                    sample_ids.append(int(row[0]))
            return max(sample_ids) + 1 if sample_ids else 1
    except:
        return 1

def parse_serial_line(line):
    """
    Parse a serial line from ESP32 into sensor values
    Format: "RAW,0,4046,3409,3721,3479,3581,4296,367,-442,210,-27,-32"
    """
    try:
        parts = line.strip().split(',')
        if len(parts) != 13:
            return None
        
        # Check if it's a RAW line
        if parts[0] != 'RAW':
            return None
        
        timestep = int(parts[1])
        values = list(map(float, parts[2:]))
        
        if len(values) != 11:
            return None
        
        return {
            'timestep': timestep,
            'flex1': values[0],
            'flex2': values[1],
            'flex3': values[2],
            'flex4': values[3],
            'flex5': values[4],
            'accel_x': values[5],
            'accel_y': values[6],
            'accel_z': values[7],
            'gyro_x': values[8],
            'gyro_y': values[9],
            'gyro_z': values[10]
        }
    except (ValueError, IndexError):
        return None

def collect_sample(ser, sample_id, label, output_file, timeout=10):
    """
    Collect one sample (20 timesteps) from the serial port
    Returns: (success, rows_collected)
    """
    buffer = []
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Collecting sample {sample_id} for label: '{label}'")
    print("Press the button on the glove to start...")
    print(f"{'='*50}")
    
    # Wait for capture to start
    while time.time() - start_time < timeout:
        raw = ser.readline()
        if not raw:
            continue
        
        line = raw.decode('utf-8', errors='ignore').strip()
        
        # Check for capture start
        if line.startswith('CAPTURING'):
            print("  Capture started...")
            buffer = []
            continue
        
        # Parse data line
        if line.startswith('RAW,'):
            data = parse_serial_line(line)
            if data:
                buffer.append(data)
                progress = len(buffer) / NUM_TIMESTEPS * 100
                print(f"  Timestep {len(buffer)}/{NUM_TIMESTEPS} ({progress:.0f}%)", end='\r')
                
                if len(buffer) == NUM_TIMESTEPS:
                    print(f"\n  ✓ Capture complete! {len(buffer)} timesteps collected")
                    break
        
        # Check for sample complete
        if line == 'SAMPLE_DONE':
            print("  Sample completed by ESP32")
            break
        
        # Status messages
        if line and not line.startswith('RAW'):
            print(f"[ESP32] {line}")
    
    # Check if we have enough timesteps
    if len(buffer) < NUM_TIMESTEPS:
        print(f"  ✗ Error: Only {len(buffer)}/{NUM_TIMESTEPS} timesteps collected")
        return False, len(buffer)
    
    # Write to CSV
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if os.path.getsize(output_file) == 0:
            writer.writerow(HEADER)
        
        # Write each timestep
        for row in buffer:
            writer.writerow([
                sample_id,
                label,
                row['timestep'],
                row['flex1'],
                row['flex2'],
                row['flex3'],
                row['flex4'],
                row['flex5'],
                row['accel_x'],
                row['accel_y'],
                row['accel_z'],
                row['gyro_x'],
                row['gyro_y'],
                row['gyro_z']
            ])
            f.flush()
    
    print(f"  ✓ Saved to {output_file}")
    return True, len(buffer)

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='BaSL Glove Data Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect a single sample
  python data_collection.py --port COM6 --label salam --output dataset.csv
  
  # Collect multiple samples with different labels
  python data_collection.py --port COM6 --label apni --output dataset.csv
  
  # After collection, you can merge multiple CSV files:
  # Your training script will handle merging
        """
    )
    
    parser.add_argument('--port', required=True, help='Serial port (e.g., COM6)')
    parser.add_argument('--label', required=True, help='Label for this sign (e.g., salam, apni)')
    parser.add_argument('--output', default='dataset.csv', help='Output CSV file (default: dataset.csv)')
    parser.add_argument('--baud', default=115200, type=int, help='Baud rate (default: 115200)')
    parser.add_argument('--samples', default=1, type=int, help='Number of samples to collect (default: 1)')
    
    args = parser.parse_args()
    
    # Connect to serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Connected to {args.port} at {args.baud} baud.")
    except serial.SerialException as e:
        print(f"Error connecting to {args.port}: {e}")
        return
    
    # Get next sample ID
    sample_id = get_next_sample_id(args.output)
    print(f"Next sample_id: {sample_id}")
    print(f"Label: {args.label}")
    print(f"Output: {args.output}")
    
    # Check if file exists and has correct header
    if os.path.isfile(args.output):
        with open(args.output, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('sample_id'):
                print("Warning: Existing file doesn't have correct header. Appending with header.")
    
    # Collect samples
    successful = 0
    for i in range(args.samples):
        print(f"\n--- Sample {i+1}/{args.samples} ---")
        success, rows = collect_sample(ser, sample_id + i, args.label, args.output)
        if success:
            successful += 1
        time.sleep(0.5)  # Small delay between samples
    
    # Summary
    print(f"\n{'='*50}")
    print("COLLECTION SUMMARY")
    print(f"{'='*50}")
    print(f"  Successful: {successful}/{args.samples}")
    print(f"  Output file: {os.path.abspath(args.output)}")
    print(f"  Last sample_id: {sample_id + successful - 1}")
    
    ser.close()
    print("Done!")

if __name__ == "__main__":
    main()