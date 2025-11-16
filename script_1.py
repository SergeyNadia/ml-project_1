#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import config
from model_pipeline import MLPipeline
from inference import InferenceEngine
from monitoring import ModelMonitor

def main():
    """Основной скрипт запуска ML pipeline"""
    print("🚀 ЗАПУСК ML PIPELINE")
    
    try:
        # 1. Загрузка данных
        print("📥 Загрузка данных...")
        data = pd.read_parquet('data/dataset.parquet')
        print(f"✅ Данные загружены: {data.shape}")
        
        # 2. Подготовка и обучение
        pipeline = MLPipeline()
        pipeline.prepare_data(data)
        pipeline.train_model()
        
        # 3. Кросс-валидация
        pipeline.cross_validation(cv=2)
        
        # 4. Оценка модели
        metrics = pipeline.evaluate_model()
        pipeline.create_evaluation_plots()
        
        # 5. Сохранение модели и препроцессора
        pipeline.save_model()
        
        # 6. Тест инференса
        inference = InferenceEngine()
        
        # Пример предсказания для одной записи
        sample_data = pipeline.X_test.iloc[0:1]
        prediction = inference.predict(sample_data)
        print(f"\n🔮 Пример предсказания: {prediction}")
        
        # Пакетное предсказание
        batch_results = inference.batch_predict(pipeline.X_test.head(10))
        print(f"\n📦 Пакетные предсказания (первые 10):")
        print(batch_results)
        
        # 7. Мониторинг
        monitor = ModelMonitor()
        monitor.save_metrics(metrics)
        monitor.performance_dashboard()
        
        print("\n✅ ML PIPELINE УСПЕШНО ЗАВЕРШЕН")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()