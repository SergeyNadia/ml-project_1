# api.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import time
import logging
from script_1 import DataPreprocessor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Глобальная переменная для движка
inference_engine = None

# АБСОЛЮТНЫЕ пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.pkl')

logger.info(f"🔍 Пути к файлам:")
logger.info(f"   Модель: {MODEL_PATH}")
logger.info(f"   Препроцессор: {PREPROCESSOR_PATH}")
logger.info(f"   Существует модель: {os.path.exists(MODEL_PATH)}")
logger.info(f"   Существует препроцессор: {os.path.exists(PREPROCESSOR_PATH)}")

class SimpleInferenceEngine:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Загрузка модели и препроцессора"""
        try:
            logger.info(f"🔄 Проверка файлов...")
            logger.info(f"   MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
            logger.info(f"   PREPROCESSOR_PATH exists: {os.path.exists(PREPROCESSOR_PATH)}")
            
            if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
                logger.info("🔄 Загрузка модели и препроцессора...")
                
                # Загружаем модель
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Модель загружена: {type(self.model).__name__}")
                
                # Загружаем препроцессор
                with open(PREPROCESSOR_PATH, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info(f"✅ Препроцессор загружен: {type(self.preprocessor).__name__}")
                
                # Проверяем атрибуты модели
                if hasattr(self.model, 'n_features_in_'):
                    logger.info(f"📊 Модель ожидает {self.model.n_features_in_} признаков")
                
            else:
                missing_files = []
                if not os.path.exists(MODEL_PATH):
                    missing_files.append(MODEL_PATH)
                if not os.path.exists(PREPROCESSOR_PATH):
                    missing_files.append(PREPROCESSOR_PATH)
                    
                logger.error(f"❌ Файлы не найдены: {missing_files}")
                logger.error("💡 Решение: Запустите script_1.py для обучения модели")
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def predict(self, data):
        """Предсказание для данных"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Модель или препроцессор не загружены")
        
        try:
            # Преобразование входных данных в DataFrame
            if isinstance(data, dict):
                X_df = pd.DataFrame([data])
            elif isinstance(data, list):
                X_df = pd.DataFrame(data)
            else:
                X_df = data
                
            logger.info(f"📊 Входные данные: {X_df.shape}")
            logger.info(f"📋 Колонки: {list(X_df.columns)}")
                
            # Преобразование данных
            X_processed = self.preprocessor.transform(X_df)
            logger.info(f"🔧 Данные после препроцессинга: {X_processed.shape}")
            
            # Предсказание
            prediction = self.model.predict(X_processed)
            probability = self.model.predict_proba(X_processed)
            
            # Формируем результат
            if len(prediction) == 1:
                # Одиночное предсказание
                return {
                    'prediction': int(prediction[0]),
                    'probability_class_0': float(probability[0][0]),
                    'probability_class_1': float(probability[0][1]),
                    'confidence': float(np.max(probability[0]))
                }
            else:
                # Пакетное предсказание
                results = []
                for i in range(len(prediction)):
                    results.append({
                        'prediction': int(prediction[i]),
                        'probability_class_0': float(probability[i][0]),
                        'probability_class_1': float(probability[i][1]),
                        'confidence': float(np.max(probability[i]))
                    })
                return results
                
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

def initialize_engine():
    """Инициализация движка"""
    global inference_engine
    logger.info("🔄 Инициализация модели...")
    inference_engine = SimpleInferenceEngine()
    
    if inference_engine.model is not None and inference_engine.preprocessor is not None:
        logger.info("✅ Модель успешно инициализирована")
        return True
    else:
        logger.error("❌ Не удалось инициализировать модель")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if inference_engine and inference_engine.model is not None:
        return jsonify({
            'status': 'healthy', 
            'model_loaded': True,
            'model_type': type(inference_engine.model).__name__,
            'message': 'API and model are working correctly'
        })
    else:
        return jsonify({
            'status': 'degraded', 
            'model_loaded': False,
            'message': 'Model not loaded. Please train model first using script_1.py'
        }), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для предсказаний"""
    if inference_engine is None or inference_engine.model is None:
        return jsonify({
            'error': 'Model not loaded', 
            'message': 'Please train model first using script_1.py',
            'solution': 'Run: python script_1.py to train the model'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"📥 Получены данные для предсказания: {data}")
        
        # Предсказание
        result = inference_engine.predict(data)
        logger.info(f"✅ Предсказание завершено: {result}")
        
        return jsonify({
            'result': result,
            'status': 'success'
        })
            
    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Информация о модели"""
    if inference_engine is None or inference_engine.model is None:
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
        
        # Добавляем информацию о фичах если доступно
        if hasattr(inference_engine.model, 'n_features_in_'):
            info['features_count'] = inference_engine.model.n_features_in_
        
        if hasattr(inference_engine.model, 'n_estimators'):
            info['n_estimators'] = inference_engine.model.n_estimators
            
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Главная страница"""
    model_status = "loaded" if inference_engine and inference_engine.model else "not loaded"
    
    return jsonify({
        'message': 'ML API Server',
        'model_status': model_status,
        'status': 'API is running' if model_status == 'loaded' else 'API running but model not loaded',
        'endpoints': {
            'GET /': 'Эта страница',
            'GET /health': 'Статус здоровья API и модели',
            'GET /model/info': 'Информация о загруженной модели',
            'POST /predict': 'Предсказание (требует JSON с features)'
        },
        'example_request': {
            'url': 'POST /predict',
            'body': {'feature1': 0.5, 'feature2': 1.2, 'feature3': -0.3}
        }
    })

# Инициализация при запуске
initialize_engine()

if __name__ == '__main__':
    logger.info("🚀 Starting ML API server...")
    from waitress import serve
    logger.info("🌐 API server listening on http://0.0.0.0:8000")
    serve(app, host='0.0.0.0', port=8000)