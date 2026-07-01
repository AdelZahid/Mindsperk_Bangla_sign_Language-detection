"""
BaSL Glove — Input Collection Module
Collects 20 timesteps of raw sensor data from the glove via serial port

Usage:
    python input_collect.py --port COM6 --output raw_data.npy

Returns:
    raw_data.npy: numpy array of shape (20, 11) with raw sensor values
"""

import serial
import numpy as np
import argparse
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONSTANTS
# ============================================

NUM_TIMESTEPS = 20
NUM_FEATURES = 11
TIMEOUT_SECONDS = 10  # Max time to wait for capture

FEATURE_NAMES = [
    'flex1', 'flex2', 'flex3', 'flex4', 'flex5',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z'
]

# ============================================
# SERIAL PARSING FUNCTIONS
# ============================================

def parse_serial_line(line):
    """
    Parse a serial line from ESP32 into sensor values
    Format: "RAW,0,4046,3409,3721,3479,3581,4296,367,-442,210,-27,-32"
    Returns: dict of sensor values or None
    """
    try:
        parts = line.strip().split(',')
        
        # Check if it's a RAW data line (13 parts: RAW, timestep, 11 values)
        if len(parts) != 13:
            return None
        
        if parts[0] != 'RAW':
            return None
        
        # Extract values (skip RAW and timestep)
        values = list(map(float, parts[2:]))
        
        if len(values) != 11:
            return None
        
        return {
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

def create_sample_from_buffer(buffer):
    """
    Convert buffer of 20 readings to (20, 11) numpy array
    """
    if len(buffer) < NUM_TIMESTEPS:
        return None
    
    sample = np.zeros((NUM_TIMESTEPS, NUM_FEATURES))
    for t, reading in enumerate(buffer[:NUM_TIMESTEPS]):
        sample[t] = [
            reading.get('flex1', 0),
            reading.get('flex2', 0),
            reading.get('flex3', 0),
            reading.get('flex4', 0),
            reading.get('flex5', 0),
            reading.get('accel_x', 0),
            reading.get('accel_y', 0),
            reading.get('accel_z', 0),
            reading.get('gyro_x', 0),
            reading.get('gyro_y', 0),
            reading.get('gyro_z', 0)
        ]
    
    return sample

# ============================================
# MAIN COLLECTOR CLASS
# ============================================

class DataCollector:
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.buffer = deque(maxlen=NUM_TIMESTEPS)
        self.capturing = False
        self.sample_count = 0
        self.raw_data = None
        
    def connect(self):
        """Connect to serial port"""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"✅ Connected to {self.port} at {self.baud} baud.")
            return True
        except serial.SerialException as e:
            print(f"❌ Error connecting to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected from serial port.")
    
    def collect_sample(self, timeout=TIMEOUT_SECONDS):
        """
        Collect one sample (20 timesteps) from the serial port
        Returns: (success, sample_data, timesteps_collected)
        """
        self.buffer.clear()
        self.capturing = False
        start_time = time.time()
        timesteps_collected = 0
        
        print(f"\n{'='*50}")
        print(f"📊 COLLECTING SAMPLE {self.sample_count + 1}")
        print("👉 Press the button on the glove to start...")
        print(f"{'='*50}")
        
        # Wait for capture to start or timeout
        while time.time() - start_time < timeout:
            if not self.ser:
                print("❌ Serial port not connected")
                return False, None, 0
            
            raw = self.ser.readline()
            if not raw:
                continue
            
            line = raw.decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            # Check for capture start
            if line.startswith('CAPTURING'):
                self.capturing = True
                self.buffer.clear()
                timesteps_collected = 0
                print("  ✅ Capture started...")
                continue
            
            # Parse data line
            if line.startswith('RAW,'):
                data = parse_serial_line(line)
                if data and self.capturing:
                    self.buffer.append(data)
                    timesteps_collected = len(self.buffer)
                    progress = timesteps_collected / NUM_TIMESTEPS * 100
                    print(f"  Timestep {timesteps_collected}/{NUM_TIMESTEPS} ({progress:.0f}%)", end='\r')
                    
                    if timesteps_collected == NUM_TIMESTEPS:
                        self.capturing = False
                        self.sample_count += 1
                        print(f"\n  ✅ Capture complete! {timesteps_collected} timesteps collected")
                        break
            
            # Check for sample complete
            if line == 'SAMPLE_DONE':
                self.capturing = False
                print("  ✅ Sample completed by ESP32")
                break
            
            # Status messages (print but don't process)
            if line and not line.startswith('RAW'):
                print(f"[ESP32] {line}")
        
        # Check if we have enough timesteps
        if timesteps_collected < NUM_TIMESTEPS:
            print(f"\n  ❌ Error: Only {timesteps_collected}/{NUM_TIMESTEPS} timesteps collected")
            return False, None, timesteps_collected
        
        # Create sample from buffer
        sample = create_sample_from_buffer(list(self.buffer))
        if sample is None:
            print("  ❌ Error: Failed to create sample from buffer")
            return False, None, timesteps_collected
        
        self.raw_data = sample
        return True, sample, timesteps_collected
    
    def collect_multiple(self, num_samples=1, save_path=None):
        """
        Collect multiple samples
        Returns: list of samples
        """
        samples = []
        for i in range(num_samples):
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            success, sample, count = self.collect_sample()
            if success and sample is not None:
                samples.append(sample)
                print(f"  ✅ Sample {i+1} collected successfully")
            else:
                print(f"  ❌ Sample {i+1} failed")
            
            # Small delay between samples
            if i < num_samples - 1:
                time.sleep(0.5)
        
        # Save if path provided
        if save_path and samples:
            np.save(save_path, np.array(samples))
            print(f"\n💾 Saved {len(samples)} samples to: {save_path}")
        
        return samples

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='BaSL Glove Data Collector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect a single sample
  python input_collect.py --port COM6
  
  # Collect 5 samples and save
  python input_collect.py --port COM6 --samples 5 --output raw_data.npy
  
  # Collect and return (for use in other scripts)
  python input_collect.py --port COM6 --return-only
        """
    )
    
    parser.add_argument('--port', required=True, help='Serial port (e.g., COM6)')
    parser.add_argument('--baud', default=115200, type=int, help='Baud rate (default: 115200)')
    parser.add_argument('--samples', default=1, type=int, help='Number of samples to collect')
    parser.add_argument('--output', help='Output .npy file to save raw data')
    parser.add_argument('--return-only', action='store_true', 
                       help='Only return data without saving (for integration)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("🧤 BaSL Glove - Data Collector")
    print("="*50)
    
    # Create collector
    collector = DataCollector(args.port, args.baud)
    
    if not collector.connect():
        return
    
    try:
        # Collect samples
        samples = collector.collect_multiple(
            num_samples=args.samples,
            save_path=args.output if not args.return_only else None
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print("📊 COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"  Samples collected: {len(samples)}")
        if samples:
            print(f"  Sample shape: {samples[0].shape}")
            print(f"  Features: {NUM_FEATURES}")
            print(f"  Timesteps: {NUM_TIMESTEPS}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user.")
    finally:
        collector.disconnect()

if __name__ == "__main__":
    main()