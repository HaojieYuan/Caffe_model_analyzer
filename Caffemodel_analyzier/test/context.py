"""this is used for importing modules in other directories"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import create_labellist
import model
import predictor
import transformer
import image_and_label