#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Config:
    RANDOM_STATE = 21
    TEST_SIZE = 0.2
    MODEL_PATH = 'models/best_model.pkl'
    METRICS_PATH = 'metrics_history.json'
    PREPROCESSOR_PATH = 'models/preprocessor.pkl'

config = Config()