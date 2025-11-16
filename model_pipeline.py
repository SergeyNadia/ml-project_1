#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, 
                           average_precision_score, precision_recall_curve)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from config import config
from data_preprocessor import DataPreprocessor

class MLPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics_history = []
    
    def prepare_data(self, data, target_col='target'):
        """Подготовка данных для обучения"""
        print("🛠️ Подготовка данных...")
        
        # Убираем категориальные признаки перед разделением
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Разделение на признаки и целевую переменную
        X = numeric_data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Разделение на train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Препроцессинг
        self.preprocessor = DataPreprocessor()
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"✅ Данные подготовлены: Train {self.X_train_processed.shape}, Test {self.X_test_processed.shape}")
        print(f"🎯 Целевая переменная: {y.value_counts().to_dict()}")
    
    def train_model(self, model_params=None):
        """Обучение модели с проверкой на переобучение"""
        print("🎯 Обучение модели...")
        
        # Параметры по умолчанию
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': config.RANDOM_STATE,
                'n_jobs': -1
            }
        
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(self.X_train_processed, self.y_train)
        
        # Проверка на переобучение
        self._check_overfitting()
        
        return self.model
    
    def _check_overfitting(self):
        """Проверка модели на переобучение"""
        print("\n🔍 Проверка на переобучение...")
        
        # Предсказания
        y_train_pred = self.model.predict(self.X_train_processed)
        y_test_pred = self.model.predict(self.X_test_processed)
        y_train_proba = self.model.predict_proba(self.X_train_processed)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test_processed)[:, 1]
        
        # Метрики
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        accuracy_diff = train_accuracy - test_accuracy
        
        train_auc = roc_auc_score(self.y_train, y_train_proba)
        test_auc = roc_auc_score(self.y_test, y_test_proba)
        auc_diff = train_auc - test_auc
        
        print(f"Accuracy: Train={train_accuracy:.4f}, Test={test_accuracy:.4f}, Diff={accuracy_diff:.4f}")
        print(f"ROC AUC:  Train={train_auc:.4f}, Test={test_auc:.4f}, Diff={auc_diff:.4f}")
        
        if accuracy_diff > 0.05 or auc_diff > 0.05:
            print("🚨 ВОЗМОЖНО ПЕРЕОБУЧЕНИЕ!")
            return True
        else:
            print("✅ Модель сбалансирована")
            return False
    
    def evaluate_model(self):
        """Полная оценка модели"""
        print("\n📊 Оценка модели...")
        
        y_pred = self.model.predict(self.X_test_processed)
        y_proba = self.model.predict_proba(self.X_test_processed)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # Сохраняем метрики
        self.metrics_history.append(metrics)
        
        # Вывод результатов
        print(f"🎯 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"✅ Precision: {metrics['precision']:.4f}")
        print(f"🔄 Recall:    {metrics['recall']:.4f}")
        print(f"⚖️ F1-Score:  {metrics['f1']:.4f}")
        print(f"📊 ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Детальный отчет
        print(f"\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return metrics

    def create_evaluation_plots(self):
        """Создание графиков для оценки модели"""
        print("\n📈 Создание оценочных графиков...")
        
        y_pred = self.model.predict(self.X_test_processed)
        y_proba = self.model.predict_proba(self.X_test_processed)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Матрица ошибок
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        
        # 2. ROC-кривая
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        
        # 3. Precision-Recall кривая
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        ap = average_precision_score(self.y_test, y_proba)
        axes[1,0].plot(recall, precision, label=f'AP = {ap:.3f}')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        
        # 4. Важность признаков
        feature_importance = pd.DataFrame({
            'feature': self.X_train_processed.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[1,1])
        axes[1,1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def cross_validation(self, cv=5):
        """Кросс-валидация"""
        print(f"🔄 Кросс-валидация ({cv} фолдов)...")
        cv_scores = cross_val_score(
            self.model, self.X_train_processed, self.y_train, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        print(f"✅ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return cv_scores

    def save_model(self):
        """Сохранение модели и препроцессора"""
        # Создаем папку models если её нет
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(config.PREPROCESSOR_PATH), exist_ok=True)
        
        with open(config.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(config.PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print("💾 Модель и препроцессор сохранены")