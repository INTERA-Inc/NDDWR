# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:09:05 2025

@author: shjordan
"""

from spirit_war01_model_preprocess import main_preprocess
from spirit_war02_model_build_4_layer_fullTime import main_build_model

# Pre-process the input data
main_preprocess()

# Build the model
main_build_model()

# Set up IES