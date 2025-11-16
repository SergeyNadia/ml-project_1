#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
    
    def identify_features(self, data):
        """Идентификация типов признаков"""
        self.numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = []  # Убираем все категориальные признаки
            
        print(f"🔢 Числовые признаки ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"📝 Категориальные признаки ({len(self.categorical_features)}): {self.categorical_features}")
        
        return self.numeric_features, self.categorical_features
    
    def fit_transform(self, data):
        """Обучение и преобразование данных"""
        self.identify_features(data)
        
        # Создаем препроцессор ТОЛЬКО для числовых признаков
        numeric_transformer = StandardScaler()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features)
            ],
            remainder='drop'  # Удаляем все остальные столбцы
        )
        
        # Применяем преобразования
        transformed_data = self.preprocessor.fit_transform(data)
        
        # Создаем DataFrame с преобразованными данными
        feature_names = [f"num_{col}" for col in self.numeric_features]
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
        
        return transformed_df
    
    def transform(self, data):
        """Преобразование новых данных"""
        if self.preprocessor is None:
            raise ValueError("Сначала нужно вызвать fit_transform")
        
        transformed_data = self.preprocessor.transform(data)
        feature_names = [f"num_{col}" for col in self.numeric_features]
        
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
        return transformed_df