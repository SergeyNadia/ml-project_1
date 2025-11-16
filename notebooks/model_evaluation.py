import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, confusion_matrix
)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, results_df=None):
    """
    Оценивает модель и возвращает DataFrame с метриками
    
    Parameters:
    - model: обученная модель
    - X_train, X_test: признаки
    - y_train, y_test: целевые переменные
    - model_name: название модели
    - results_df: существующий DataFrame для добавления результатов
    
    Returns:
    - results_df: DataFrame с метриками
    """
    
    # Предсказания
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Вычисление метрик для train и test
    metrics = {
        'Model': model_name,
        'Test_Accuracy': accuracy_score(y_test, y_test_pred),
        'Test_Precision': precision_score(y_test, y_test_pred),
        'Test_Recall': recall_score(y_test, y_test_pred),
        'Test_F1': f1_score(y_test, y_test_pred),
        'Test_ROC_AUC': roc_auc_score(y_test, y_test_proba),
        'Test_AP': average_precision_score(y_test, y_test_proba),
    }
    
    # Создание или обновление DataFrame
    if results_df is None:
        results_df = pd.DataFrame([metrics])
    else:
        new_row = pd.DataFrame([metrics])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    return results_df

def confusion_matrix_graf(model, X_train, X_test, y_train, y_test, model_name):
    """
    Визуализирует матрицы ошибок для тренировочной и тестовой выборок
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    cm_train = confusion_matrix(y_train, model.predict(X_train))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Confusion Matrix - Train {model_name}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    cm_test = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Confusion Matrix - Test {model_name}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def roc_curves_graf(X_train, X_test, y_train, y_test, model, model_name):
    """
    Визуализирует ROC-кривые для тренировочной и тестовой выборок
    """
    plt.figure(figsize=(10, 8))
    
    # ROC для train
    fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1])
    roc_auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    
    # ROC для test
    fpr_test, tpr_test, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, 
             label=f'Train ROC (AUC = {roc_auc_train:.4f})')
    plt.plot(fpr_test, tpr_test, color='red', lw=2, 
             label=f'Test ROC (AUC = {roc_auc_test:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label=f'{model_name}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Train vs Test')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def precision_recall_curves_graf(X_test, X_train, y_train, y_test, model, model_name):
    """
    Визуализирует Precision-Recall кривые для тренировочной и тестовой выборок
    """
    plt.figure(figsize=(10, 8))
    
    # Precision-Recall для train
    precision_train, recall_train, _ = precision_recall_curve(y_train, model.predict_proba(X_train)[:, 1])
    ap_train = average_precision_score(y_train, model.predict_proba(X_train)[:, 1])
    
    # Precision-Recall для test
    precision_test, recall_test, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    ap_test = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    
    plt.plot(recall_train, precision_train, color='blue', lw=2, 
             label=f'Train PR (AP = {ap_train:.4f})')
    plt.plot(recall_test, precision_test, color='red', lw=2, 
             label=f'Test PR (AP = {ap_test:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - Train vs Test {model_name}')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def evaluate_all_metrics(model, X_train, X_test, y_train, y_test, model_name, results_df=None):
    """
    Комплексная оценка модели со всеми визуализациями
    """
    print(f"=== Оценка модели: {model_name} ===")
    
    confusion_matrix_graf(model, X_train, X_test, y_train, y_test, model_name)
    
    roc_curves_graf(X_train, X_test, y_train, y_test, model, model_name)
    
    precision_recall_curves_graf(X_test, X_train, y_train, y_test, model, model_name)
    
    results_df = evaluate_model(model, X_train, X_test, y_train, y_test, model_name, results_df)
    
    return results_df