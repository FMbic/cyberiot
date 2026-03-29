import os
import re
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings('ignore')

# Настройка стилей для визуализаций
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

# ============================================================================
# КОНФИГУРАЦИЯ СКРИПТА
# ============================================================================

# Поиск корня проекта
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "Deepseek-V3").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
    if PROJECT_ROOT == PROJECT_ROOT.parent:  # Достигли корня файловой системы
        break

# Директория для результатов
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class Config:
    """Глобальные конфигурационные параметры"""

    # Пути к данным
    DATA_DIR = PROJECT_ROOT / 'Deepseek-V3' / 'inSDN'
    FILES = {
        'normal': 'Normal_data.csv',
        'ovs': 'OVS.csv',
        'metasploitable': 'metasploitable-2.csv'
    }

    # Параметры загрузки
    TEST_ROWS = None  # Установите число, например 50000, для тестового запуска

    # Параметры предобработки
    COLUMNS_TO_DROP = [
        # Сетевые идентификаторы (могут привести к переобучению)
        'Flow ID', 'Flow_Id', 'flow_id',
        'Src IP', 'Src_IP', 'src_ip', 'Source IP',
        'Dst IP', 'Dst_IP', 'dst_ip', 'Destination IP',
        'Src Port', 'Src_Port', 'src_port',
        'Dst Port', 'Dst_Port', 'dst_port',
        'Timestamp', 'timestamp', 'Flow Start Time',
        # Дубликаты или нерелевантные колонки
        'Unnamed: 0', 'index'
    ]

    # Возможные названия целевой колонки (в порядке приоритета)
    TARGET_COLUMN_CANDIDATES = [
        'Label', 'label', 'Type', 'type', 'Attack', 'attack',
        'Class', 'class', 'Category', 'category', 'Traffic Type'
    ]

    # Названия нормального трафика для объединения
    BENIGN_LABELS = ['Benign', 'Normal', 'benign', 'normal', 'BENIGN', 'NORMAL']

    # Параметры моделей
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Параметры SMOTE
    SMOTE_K_NEIGHBORS = 5

    # Параметры Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20

    # Параметры XGBoost
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1


# ============================================================================
# ФУНКЦИИ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ ДАННЫХ
# ============================================================================

def detect_target_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Автоматически определяет название целевой колонки в датафрейме.
    """
    columns_lower = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    # Если не найдено, ищем колонку с небольшим числом уникальных значений
    for col in df.columns:
        if df[col].nunique() < 20 and df[col].dtype == 'object':
            print(f"⚠️  Предположительно целевая колонка: '{col}'")
            return col

    return None


def load_and_merge_datasets(
        config: Config,
        nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Загружает и объединяет три CSV-файла датасета InSDN.
    """
    dataframes = []

    for key, filename in config.FILES.items():
        # ✅ Используем pathlib для построения пути
        filepath = config.DATA_DIR / filename

        if not filepath.exists():
            print(f"❌ Файл не найден: {filepath}")
            continue

        print(f"📥 Загрузка {filename}...")

        # ✅ Явное преобразование в str для pd.read_csv
        df = pd.read_csv(
            str(filepath),
            nrows=config.TEST_ROWS if config.TEST_ROWS else nrows,
            low_memory=False
        )

        # Добавляем метку источника (опционально)
        df['source_file'] = key

        # Определяем целевую колонку
        target_col = detect_target_column(df, config.TARGET_COLUMN_CANDIDATES)
        if target_col and key == 'normal':
            print(f"   ✓ Целевая колонка в {filename}: '{target_col}'")
            # Для Normal_data.csv принудительно устанавливаем метку как 'Benign'
            if target_col:
                df[target_col] = 'Benign'

        dataframes.append(df)
        print(f"   ✓ Загружено: {len(df)} строк, {len(df.columns)} колонок")

    if not dataframes:
        raise ValueError("Не удалось загрузить ни один файл!")

    # Проверяем согласованность колонок
    reference_cols = set(dataframes[0].columns)
    for i, df in enumerate(dataframes[1:], 1):
        if set(df.columns) != reference_cols:
            print(f"⚠️  Файл {list(config.FILES.values())[i]} имеет отличающиеся колонки")
            missing = reference_cols - set(df.columns)
            extra = set(df.columns) - reference_cols
            for col in missing:
                df[col] = np.nan
            df = df.drop(columns=list(extra))

    # Объединяем все датафреймы
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    print(f"\n✅ Объединённый датасет: {merged_df.shape[0]} строк, {merged_df.shape[1]} колонок")

    return merged_df


def preprocess_data(
        df: pd.DataFrame,
        config: Config,
        target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Выполняет полную предобработку данных.
    """
    df = df.copy()
    print("\n🔧 Начинаем предобработку данных...")

    # 1. Удаляем колонки из списка исключений
    cols_to_drop = [c for c in config.COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        print(f"   🗑️  Удаляем колонки: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 2. Удаляем колонки с одним уникальным значением или полностью пустые
    n_before = len(df.columns)
    df = df.loc[:, df.nunique() > 1]
    df = df.dropna(axis=1, how='all')
    print(f"   🗑️  Удалено константных/пустых колонок: {n_before - len(df.columns)}")

    # 3. Обрабатываем бесконечные значения
    inf_cols = df.select_dtypes(include=[np.number]).columns
    df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
    print(f"   ♾️  Заменено ±inf на NaN в {len(inf_cols)} числовых колонках")

    # 4. Заполняем пропуски нулями
    df = df.fillna(0)
    print(f"   🔢 Заполнено {df.isna().sum().sum()} пропущенных значений нулями")

    # 5. Извлекаем целевую переменную
    if target_col not in df.columns:
        raise ValueError(f"Целевая колонка '{target_col}' не найдена после предобработки!")

    y_raw = df[target_col].copy()
    df = df.drop(columns=[target_col])

    # 6. Создаём бинарную метку: 'Attack' vs 'Benign'
    y_binary = y_raw.apply(
        lambda x: 'Benign' if str(x) in config.BENIGN_LABELS else 'Attack'
    )

    # 7. Создаём многоклассовую метку (стандартизируем названия)
    def standardize_label(label):
        label = str(label).strip()
        if label in config.BENIGN_LABELS:
            return 'Benign'
        label_map = {
            'brute-force-attack': 'BruteForce',
            'Web_attack': 'WebAttack',
            'Exploitation (R2L)': 'Exploitation',
            'R2L': 'Exploitation',
            'U2R': 'PrivilegeEscalation'
        }
        return label_map.get(label, label.title().replace('-', '').replace('_', ''))

    y_multiclass = y_raw.apply(standardize_label)

    print(f"   📊 Распределение классов (бинарное):\n{y_binary.value_counts()}")
    print(f"   📊 Распределение классов (многоклассовое):\n{y_multiclass.value_counts()}")

    # 8. Удаляем нечисловые признаки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(df.columns):
        dropped = set(df.columns) - set(numeric_cols)
        print(f"   🗑️  Удалено нечисловых признаков: {dropped}")
        df = df[numeric_cols]

    # 9. Проверяем на наличие бесконечных значений
    if np.isinf(df.values).any():
        print("⚠️  Обнаружены остаточные inf значения, заменяем на максимальное конечное")
        df = df.replace([np.inf, -np.inf], df[np.isfinite(df)].max().max())

    return df, y_binary, y_multiclass


# ============================================================================
# ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ МОДЕЛЕЙ
# ============================================================================

def train_test_split_with_smote(
        X: pd.DataFrame,
        y: pd.Series,
        config: Config,
        apply_smote: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет данные на обучающую/тестовую выборки и применяет SMOTE.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    print(f"\n📐 Разделение данных:")
    print(f"   Train: {len(X_train)} строк")
    print(f"   Test: {len(X_test)} строк")

    # Применение SMOTE только к обучающей выборке
    if apply_smote and y_train.nunique() == 2:
        print(f"\n⚖️  Применяем SMOTE (k={config.SMOTE_K_NEIGHBORS}) к обучающей выборке...")
        smote = SMOTE(
            k_neighbors=config.SMOTE_K_NEIGHBORS,
            random_state=config.RANDOM_STATE
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   ✓ После SMOTE: {len(X_train)} строк")
        print(f"   📊 Новое распределение:\n{y_train.value_counts()}")

    return X_train, X_test, y_train, y_test


def train_models(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: Config
) -> Dict[str, Dict]:
    """
    Обучает Random Forest и XGBoost модели.

    Returns:
        Словарь с обученными моделями и энкодерами
    """
    models = {}

    # ========================================================================
    # Random Forest (работает со строковыми метками)
    # ========================================================================
    print(f"\n🌲 Обучаем Random Forest (n_estimators={config.RF_N_ESTIMATORS})...")
    rf_model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = {
        'model': rf_model,
        'encoder': None
    }
    print("   ✓ Random Forest обучен")

    # ========================================================================
    # XGBoost (требует числовые метки)
    # ========================================================================
    try:
        import xgboost as xgb
        print(f"\n🚀 Обучаем XGBoost (n_estimators={config.XGB_N_ESTIMATORS})...")

        # 🔑 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Кодирование меток
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        print(
            f"   📊 Классы закодированы: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        xgb_model = xgb.XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            random_state=config.RANDOM_STATE,
            eval_metric='mlogloss' if len(label_encoder.classes_) > 2 else 'logloss',
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train_encoded)

        models['XGBoost'] = {
            'model': xgb_model,
            'encoder': label_encoder
        }
        print("   ✓ XGBoost обучен")

    except ImportError:
        print("⚠️  XGBoost не установлен, пропускаем...")

    return models


def evaluate_model(
        model_info: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        scenario: str = 'binary'
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Оценивает качество модели и выводит отчёт.
    """
    print(f"\n🔍 Оценка модели {model_name} ({scenario}):")
    print("=" * 60)

    model = model_info['model']
    encoder = model_info['encoder']

    # Предсказания
    y_pred_encoded = model.predict(X_test)

    # 🔑 Декодируем предсказания XGBoost обратно в строки
    if encoder is not None:
        y_pred = encoder.inverse_transform(y_pred_encoded)
        y_test_decoded = encoder.inverse_transform(encoder.transform(y_test))
    else:
        y_pred = y_pred_encoded
        y_test_decoded = y_test

    # Основные метрики
    metrics = {
        'accuracy': accuracy_score(y_test_decoded, y_pred),
        'precision': precision_score(y_test_decoded, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test_decoded, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test_decoded, y_pred, average='weighted', zero_division=0)
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")

    # Подробный classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test_decoded, y_pred, zero_division=0))

    return metrics, y_pred


def plot_confusion_matrix(
        y_true: pd.Series,
        y_pred: pd.Series,
        classes: List[str],
        model_name: str,
        scenario: str,
        save_path: Optional[Path] = None
):
    """
    Строит тепловую карту матрицы ошибок.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name} ({scenario})', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Матрица ошибок сохранена: {save_path}")

    plt.show()


def plot_feature_importance(
        model,
        feature_names: List[str],
        model_name: str,
        top_n: int = 20,
        save_path: Optional[Path] = None
):
    """
    Визуализирует топ-N наиболее важных признаков.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("⚠️  Модель не поддерживает feature_importances_")
        return

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feat_imp,
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=11)
    plt.ylabel('Feature Name', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Feature importance сохранён: {save_path}")

    plt.show()

    print(f"\n🏆 Топ-{top_n} наиболее важных признаков ({model_name}):")
    for i, row in feat_imp.iterrows():
        print(f"   {i + 1:2d}. {row['feature']:<40} {row['importance']:.4f}")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная точка входа в скрипт"""
    print("\n" + "=" * 80)
    print("🛡️  КЛАССИФИКАТОР ОБНАРУЖЕНИЯ АТАК InSDN - ЗАПУСК")
    print("=" * 80)

    config = Config()

    # ========================================================================
    # 🔍 Проверка доступности файлов
    # ========================================================================
    print("\n🔍 Проверка путей:")
    for key, filename in config.FILES.items():
        filepath = config.DATA_DIR / filename
        exists = "✓" if filepath.exists() else "❌"
        print(f"   {exists} {filename:<30} → {filepath}")

    if not all((config.DATA_DIR / f).exists() for f in config.FILES.values()):
        print("\n⚠️  Исправьте пути в Config.DATA_DIR")
        return

    # ========================================================================
    # ШАГ 1: Загрузка и объединение данных
    # ========================================================================
    print("\n📁 ШАГ 1: Загрузка данных")
    print("-" * 40)

    df = load_and_merge_datasets(config, nrows=Config.TEST_ROWS)

    target_col = detect_target_column(df, config.TARGET_COLUMN_CANDIDATES)
    if not target_col:
        raise ValueError("❌ Не удалось автоматически определить целевую колонку!")
    print(f"✅ Целевая колонка: '{target_col}'")

    # ========================================================================
    # ШАГ 2: Предобработка данных
    # ========================================================================
    print("\n🔧 ШАГ 2: Предобработка")
    print("-" * 40)

    X, y_binary, y_multiclass = preprocess_data(df, config, target_col)

    print(f"✅ Готово! Признаков: {X.shape[1]}, Примеров: {X.shape[0]}")

    # ========================================================================
    # ШАГ 3: Бинарная классификация (Attack vs Benign)
    # ========================================================================
    print("\n🎯 ШАГ 3: Бинарная классификация")
    print("=" * 60)

    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split_with_smote(
        X, y_binary, config, apply_smote=True
    )

    models_bin = train_models(X_train_bin, y_train_bin, config)

    results_binary = {}
    for name, model_info in models_bin.items():
        metrics, y_pred_bin = evaluate_model(
            model_info, X_test_bin, y_test_bin, name, scenario='binary'
        )
        results_binary[name] = {
            'metrics': metrics,
            'predictions': y_pred_bin,
            'model': model_info['model'],
            'encoder': model_info['encoder']
        }

        classes_bin = ['Benign', 'Attack']
        plot_confusion_matrix(
            y_test_bin, y_pred_bin, classes_bin, name, 'binary',
            save_path=RESULTS_DIR / f'confusion_matrix_{name}_binary.png'
        )

    # ========================================================================
    # ШАГ 4: Многоклассовая классификация
    # ========================================================================
    print("\n🎯 ШАГ 4: Многоклассовая классификация")
    print("=" * 60)

    X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split_with_smote(
        X, y_multiclass, config, apply_smote=False
    )

    models_mc = train_models(X_train_mc, y_train_mc, config)

    results_multiclass = {}
    for name, model_info in models_mc.items():
        metrics, y_pred_mc = evaluate_model(
            model_info, X_test_mc, y_test_mc, name, scenario='multiclass'
        )
        results_multiclass[name] = {
            'metrics': metrics,
            'predictions': y_pred_mc,
            'model': model_info['model'],
            'encoder': model_info['encoder']
        }

        classes_mc = sorted(y_multiclass.unique())
        plot_confusion_matrix(
            y_test_mc, y_pred_mc, classes_mc, name, 'multiclass',
            save_path=RESULTS_DIR / f'confusion_matrix_{name}_multiclass.png'
        )

    # ========================================================================
    # ШАГ 5: Сравнение моделей и визуализация важности признаков
    # ========================================================================
    print("\n📊 ШАГ 5: Сравнение и визуализация")
    print("=" * 60)

    best_model_name = max(
        results_binary.keys(),
        key=lambda k: results_binary[k]['metrics']['f1']
    )
    best_model = results_binary[best_model_name]['model']

    print(f"🏆 Лучшая модель (бинарная): {best_model_name}")
    print(f"   F1-Score: {results_binary[best_model_name]['metrics']['f1']:.4f}")

    plot_feature_importance(
        best_model,
        X.columns.tolist(),
        best_model_name,
        top_n=20,
        save_path=RESULTS_DIR / f'feature_importance_{best_model_name}.png'
    )

    # Сводная таблица результатов
    print(f"\n📋 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("-" * 70)
    print(f"{'Модель':<15} {'Сценарий':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 70)

    for name, res in results_binary.items():
        m = res['metrics']
        print(
            f"{name:<15} {'Binary':<12} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f}")

    for name, res in results_multiclass.items():
        m = res['metrics']
        print(
            f"{name:<15} {'Multiclass':<12} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f}")

    print("-" * 70)

    # ========================================================================
    # ЗАВЕРШЕНИЕ
    # ========================================================================
    print("\n✅ Работа скрипта завершена!")
    print(f"💾 Результаты сохранены в: {RESULTS_DIR.resolve()}")
    print("\n💡 Рекомендации:")
    print("   • Для production используйте кросс-валидацию")
    print("   • Настройте гиперпараметры через GridSearch/Optuna")
    print("   • Рассмотрите ансамблирование моделей")
    print("   • Для онлайн-детекции добавьте мониторинг дрейфа данных")

    return results_binary, results_multiclass


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    binary_results, multiclass_results = main()