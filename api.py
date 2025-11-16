#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import logging
import os
from inference import InferenceEngine
from config import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Глобальная переменная для движка
inference_engine = None

def initialize_engine():
    """Инициализация движка для предсказаний"""
    global inference_engine
    logger.info("🔄 Инициализация модели...")
    
    try:
        inference_engine = InferenceEngine()
        
        if inference_engine.model is not None and inference_engine.preprocessor is not None:
            logger.info("✅ Модель успешно загружена")
            return True
        else:
            logger.error("❌ Не удалось загрузить модель или препроцессор")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if inference_engine and inference_engine.model else 'degraded',
        'model_loaded': bool(inference_engine and inference_engine.model),
        'message': 'API is working correctly'
    }
    
    if inference_engine and inference_engine.model:
        status.update({
            'model_type': type(inference_engine.model).__name__,
            'preprocessor_type': type(inference_engine.preprocessor).__name__
        })
    else:
        status['message'] = 'Model not loaded. Please train model first using main.py'
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для предсказаний"""
    if not inference_engine or not inference_engine.model:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train model first using main.py'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"📥 Получены данные для предсказания")
        
        # Предсказание
        result = inference_engine.predict(data)
        logger.info(f"✅ Предсказание завершено")
        
        return jsonify({
            'result': result,
            'status': 'success'
        })
            
    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint для пакетных предсказаний"""
    if not inference_engine or not inference_engine.model:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train model first using main.py'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of records'}), 400
        
        logger.info(f"📥 Получено {len(data)} записей для пакетного предсказания")
        
        # Пакетное предсказание
        import pandas as pd
        df = pd.DataFrame(data)
        results = inference_engine.batch_predict(df)
        
        if results is not None:
            return jsonify({
                'results': results.to_dict('records'),
                'status': 'success',
                'count': len(results)
            })
        else:
            return jsonify({'error': 'Batch prediction failed'}), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка пакетного предсказания: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Информация о загруженной модели"""
    if not inference_engine or not inference_engine.model:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train model first'
        }), 503
    
    try:
        info = {
            'model_type': type(inference_engine.model).__name__,
            'model_loaded': True,
            'preprocessor_type': type(inference_engine.preprocessor).__name__,
        }
        
        # Информация о модели
        if hasattr(inference_engine.model, 'n_features_in_'):
            info['n_features'] = inference_engine.model.n_features_in_
        
        if hasattr(inference_engine.model, 'n_estimators'):
            info['n_estimators'] = inference_engine.model.n_estimators
            
        if hasattr(inference_engine.model, 'classes_'):
            info['classes'] = inference_engine.model.classes_.tolist()
            
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения информации о модели: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Главная страница API"""
    model_status = "loaded" if inference_engine and inference_engine.model else "not loaded"
    
    endpoints = {
        'GET /': 'Информация о API',
        'GET /health': 'Статус здоровья API и модели',
        'GET /model/info': 'Информация о загруженной модели',
        'POST /predict': 'Предсказание для одной записи',
        'POST /batch_predict': 'Пакетное предсказание для нескольких записей'
    }
    
    return jsonify({
        'message': 'ML Model API Server',
        'model_status': model_status,
        'version': '1.0',
        'endpoints': endpoints,
        'example_predict': {
            'method': 'POST',
            'url': '/predict',
            'body': {'feature1': 0.5, 'feature2': 1.2, 'feature3': -0.3}
        },
        'example_batch_predict': {
            'method': 'POST', 
            'url': '/batch_predict',
            'body': [
                {'feature1': 0.5, 'feature2': 1.2, 'feature3': -0.3},
                {'feature1': 0.1, 'feature2': 0.8, 'feature3': 0.2}
            ]
        }
    })

# Инициализация при импорте
initialize_engine()

if __name__ == '__main__':
    logger.info("🚀 Starting ML API Server...")
    
    # Проверяем инициализацию
    if inference_engine and inference_engine.model:
        logger.info("✅ Model loaded successfully")
        logger.info(f"📊 Model: {type(inference_engine.model).__name__}")
        logger.info(f"🔧 Preprocessor: {type(inference_engine.preprocessor).__name__}")
    else:
        logger.warning("⚠️  Model not loaded - please train model first")
        logger.info("💡 Run: python main.py to train the model")
    
    # Запуск сервера
    from waitress import serve
    logger.info("🌐 Server starting on http://0.0.0.0:8000")
    serve(app, host='0.0.0.0', port=8000)