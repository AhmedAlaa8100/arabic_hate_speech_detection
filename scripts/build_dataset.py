#!/usr/bin/env python3
"""
Simple script to build the combined Arabic hate speech dataset.
This script runs the dataset building process with default parameters.
"""

import subprocess
import sys
import os

def main():
    """Build the combined dataset."""
    print("🚀 Building combined Arabic hate speech dataset...")
    print("=" * 60)
    
    # Check if scripts directory exists
    if not os.path.exists('scripts'):
        print("❌ Error: scripts directory not found!")
        sys.exit(1)
    
    # Run the dataset building script
    try:
        cmd = [sys.executable, 'scripts/build_combined_dataset.py']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Dataset building completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building dataset: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
