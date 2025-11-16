#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

from config import config

class InferenceEngine:
    def __init__(self, model_path=config.MODEL_PATH, preprocessor_path=config.PREPROCESSOR_PATH):
        self.model = self.load_model(model_path)
        self.preprocessor = self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path):
        """Загрузка обученной модели"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"❌ Модель {model_path} не найдена")
            return None
    
    def load_preprocessor(self, preprocessor_path):
        """Загрузка препроцессора"""
        try:
            with open(preprocessor_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"❌ Препроцессор {preprocessor_path} не найден")
            return None
    
    def predict(self, X):
        """Предсказание для новых данных"""
        if self.model is None or self.preprocessor is None:
            print("❌ Модель или препроцессор не загружены")
            return None
        
        # Преобразование данных
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocessor.transform(X)
        else:
            # Если передан словарь или список
            X_df = pd.DataFrame([X])
            X_processed = self.preprocessor.transform(X_df)
        
        # Предсказание
        prediction = self.model.predict(X_processed)
        probability = self.model.predict_proba(X_processed)
        
        return {
            'prediction': prediction[0],
            'probability_class_0': probability[0][0],
            'probability_class_1': probability[0][1],
            'confidence': np.max(probability[0])
        }

    def batch_predict(self, data):
        """Пакетное предсказание"""
        if self.model is None or self.preprocessor is None:
            print("❌ Модель или препроцессор не загружены")
            return None
        
        if not isinstance(data, pd.DataFrame):
            print("❌ Данные должны быть pandas DataFrame")
            return None
        
        print(f"🔮 Пакетное предсказание для {len(data)} записей...")
        
        # Преобразование данных
        X_processed = self.preprocessor.transform(data)
        
        # Предсказание
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1)
        })
        
        return results