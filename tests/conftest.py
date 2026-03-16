"""
sipQuant test configuration.

Shared fixtures and path setup for all test modules.
"""
import sys
import os

# Ensure sipQuant is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
