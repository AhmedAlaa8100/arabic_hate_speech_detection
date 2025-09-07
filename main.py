#!/usr/bin/env python3
"""
Main entry point for Arabic Hate Speech Detection project.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scripts.train import main

if __name__ == "__main__":
    main()