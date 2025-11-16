#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from config import config

class ModelMonitor:
    def __init__(self, metrics_path=config.METRICS_PATH):
        self.metrics_path = metrics_path
        self.metrics_history = self.load_metrics_history()
    
    def load_metrics_history(self):
        """Загрузка истории метрик"""
        try:
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_metrics(self, metrics):
        """Сохранение метрик"""
        self.metrics_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            **metrics
        })
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def check_data_drift(self, current_data, reference_data):
        """Проверка дрейфа данных"""
        print("📊 Проверка дрейфа данных...")
        
        drift_metrics = {}
        
        for col in current_data.select_dtypes(include=[np.number]).columns:
            if col in reference_data.columns:
                # KS test для числовых признаков
                from scipy.stats import ks_2samp
                stat, p_value = ks_2samp(reference_data[col].dropna(), 
                                       current_data[col].dropna())
                drift_metrics[col] = {
                    'ks_statistic': stat,
                    'p_value': p_value,
                    'has_drift': p_value < 0.05
                }
        
        drift_detected = any([m['has_drift'] for m in drift_metrics.values()])
        
        if drift_detected:
            print("🚨 Обнаружен дрейф данных!")
        else:
            print("✅ Дрейф данных не обнаружен")
        
        return drift_metrics
    
    def performance_dashboard(self):
        """Дашборд для мониторинга производительности"""
        if len(self.metrics_history) == 0:
            print("❌ Нет данных для дашборда")
            return
        
        metrics_df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Тренд accuracy
        if 'accuracy' in metrics_df.columns:
            axes[0,0].plot(metrics_df['accuracy'])
            axes[0,0].set_title('Accuracy over time')
            axes[0,0].set_ylabel('Accuracy')
        
        # Тренд ROC AUC
        if 'roc_auc' in metrics_df.columns:
            axes[0,1].plot(metrics_df['roc_auc'])
            axes[0,1].set_title('ROC AUC over time')
            axes[0,1].set_ylabel('ROC AUC')
        
        # Распределение F1-score
        if 'f1' in metrics_df.columns:
            axes[1,0].hist(metrics_df['f1'], bins=10)
            axes[1,0].set_title('F1-Score Distribution')
            axes[1,0].set_xlabel('F1-Score')
        
        # Последняя матрица ошибок
        if 'confusion_matrix' in metrics_df.columns and len(metrics_df) > 0:
            last_cm = metrics_df.iloc[-1]['confusion_matrix']
            sns.heatmap(last_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
            axes[1,1].set_title('Latest Confusion Matrix')
        
        plt.tight_layout()
        plt.show()