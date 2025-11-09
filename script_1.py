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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Конфигурация
class Config:
    RANDOM_STATE = 21
    TEST_SIZE = 0.2
    MODEL_PATH = 'models/best_model.pkl'
    # SCALER_PATH = 'scaler.pkl'
    METRICS_PATH = 'metrics_history.json'
    PREPROCESSOR_PATH = 'models/preprocessor.pkl'
    
config = Config()


class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
        self.label_encoders = {}
    
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
                # Убираем categorical_transformer полностью
            ],
            remainder='drop'  # Удаляем все остальные столбцы
        )
        
        # Применяем преобразования
        transformed_data = self.preprocessor.fit_transform(data)
        
        # Создаем DataFrame с преобразованными данными
        feature_names = [f"num_{col}" for col in self.numeric_features]  # Только числовые
        
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
        
        return transformed_df
    
    def transform(self, data):
        """Преобразование новых данных"""
        if self.preprocessor is None:
            raise ValueError("Сначала нужно вызвать fit_transform")
        
        transformed_data = self.preprocessor.transform(data)
        feature_names = (
            [f"num_{col}" for col in self.numeric_features]
            # [f"cat_{col}" for col in self.categorical_features]
        )
        
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
        return transformed_df


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
        with open(config.MODEL_PATH, 'wb') as f:
            pickle.dump(pipeline.model, f)
        with open(config.PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump(pipeline.preprocessor, f)
        print("💾 Модель и препроцессор сохранены")
        
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