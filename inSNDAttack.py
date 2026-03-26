import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ
# ============================================================================
# Укажите пути к файлам (если нужно, добавьте аргумент nrows для тестирования)
normal_path = 'inSDN/Normal_data.csv'
ovs_path = 'inSDN/OVS.csv'
metasploitable_path = 'inSDN/metasploitable-2.csv'

# Для тестового запуска на части данных можно установить nrows, например, 50000
nrows = None  # Замените на число, чтобы ограничить количество строк

print("Загрузка данных...")
df_normal = pd.read_csv(normal_path, nrows=nrows)
df_ovs = pd.read_csv(ovs_path, nrows=nrows)
df_meta = pd.read_csv(metasploitable_path, nrows=nrows)

# Добавим колонку-источник (опционально) для отслеживания происхождения данных
df_normal['source'] = 'normal'
df_ovs['source'] = 'ovs'
df_meta['source'] = 'metasploitable'

# Проверим, что во всех файлах одинаковый набор колонок
# (в датасете они одинаковы, но на всякий случай унифицируем порядок)
common_columns = set(df_normal.columns) & set(df_ovs.columns) & set(df_meta.columns)
# Удалим 'source' из проверки, так как её добавили мы
common_columns.discard('source')

df_normal = df_normal[list(common_columns) + ['source']]
df_ovs = df_ovs[list(common_columns) + ['source']]
df_meta = df_meta[list(common_columns) + ['source']]

# Объединяем
df = pd.concat([df_normal, df_ovs, df_meta], axis=0, ignore_index=True)
print(f"Объединенный датафрейм: {df.shape[0]} строк, {df.shape[1]} колонок")

# ============================================================================
# 2. ОПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ КОЛОНКИ
# ============================================================================
# Ищем колонку, которая содержит тип трафика (названия могут быть: 'Label', 'Type', 'Attack', 'category')
target_candidates = ['Label', 'Type', 'Attack', 'category', 'label', 'type', 'attack']
target_col = None
for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # Если не нашли по имени, попробуем найти колонку с уникальными строковыми значениями
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            print(f"Возможная целевая колонка: {col} со значениями {df[col].unique()[:10]}")
            target_col = col
            break

if target_col is None:
    raise ValueError("Не удалось определить целевую колонку. Проверьте названия в файлах.")

print(f"Целевая колонка: '{target_col}'")
print(f"Уникальные значения: {df[target_col].unique()}")

# ============================================================================
# 3. ПРЕДОБРАБОТКА ДАННЫХ
# ============================================================================
# Удаляем колонки, которые полностью пустые
df = df.dropna(axis=1, how='all')
print(f"Колонок после удаления полностью пустых: {df.shape[1]}")

# Удаляем колонки с одним уникальным значением (бесполезны для модели)
unique_counts = df.nunique()
single_value_cols = unique_counts[unique_counts == 1].index.tolist()
df = df.drop(columns=single_value_cols)
print(f"Удалено колонок с одним значением: {len(single_value_cols)}")

# Удаляем колонки с IP-адресами и временными метками (могут вызвать переобучение)
ip_like_cols = [col for col in df.columns if 'ip' in col.lower() or 'addr' in col.lower() or 'address' in col.lower()]
time_like_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
cols_to_drop = ip_like_cols + time_like_cols
# Убедимся, что не удаляем целевую колонку
cols_to_drop = [col for col in cols_to_drop if col != target_col]
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"Удалено колонок с IP/временем: {len(cols_to_drop)}")

# Заменяем inf и -inf на NaN
df = df.replace([np.inf, -np.inf], np.nan)
# Заполняем NaN нулями (можно также медианой, но для простоты используем 0)
df = df.fillna(0)

# Преобразуем целевую колонку в числовую (если строковая) для обработки
y_raw = df[target_col].copy()
# Оставляем оригинальные названия атак для многоклассовой классификации
attack_names = y_raw.unique()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# Бинарная метка: 0 — нормальный трафик, 1 — атака
# Нормальными считаем все, что содержит 'Normal' или 'Benign'
normal_keywords = ['normal', 'benign', 'Normal', 'Benign', 'NORMAL', 'BENIGN']
is_normal = y_raw.str.contains('|'.join(normal_keywords), na=False, case=False)
y_binary = (~is_normal).astype(int)

print(f"Распределение классов в бинарной задаче:\n{pd.Series(y_binary).value_counts()}")
print(f"Распределение типов атак:\n{pd.Series(y_raw).value_counts()}")

# ============================================================================
# 4. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================================
# Удаляем целевую колонку из признаков
X = df.drop(columns=[target_col, 'source'] if 'source' in df.columns else [target_col])
# Все признаки должны быть числовыми
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print(f"Размер матрицы признаков: {X.shape}")

# Разделение на обучающую и тестовую выборки (стратифицируем по бинарной метке)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
# Для многоклассовой классификации также разделим с оригинальными метками
_, _, y_train_multi, y_test_multi = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_binary
)

# ============================================================================
# 5. БАЛАНСИРОВКА КЛАССОВ С ПОМОЩЬЮ SMOTE (только на обучающей выборке)
# ============================================================================
print("Применяем SMOTE для балансировки классов...")

# Бинарная классификация
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Размер обучающей выборки после SMOTE: {X_train_resampled.shape[0]}")

# Многоклассовая классификация - с адаптивным k_neighbors для редких классов
print(f"\nРаспределение классов в мультиклассовой обучающей выборке:")
class_counts = Counter(y_train_multi)
print(class_counts)

# Определяем подходящее значение k_neighbors
min_samples = min(class_counts.values())
if min_samples < 6:
    # Для редких классов используем k_neighbors = min(2, min_samples-1)
    # Минимум 1, чтобы избежать ошибки
    k = max(1, min(2, min_samples - 1))
    print(f"Внимание: обнаружен редкий класс с {min_samples} образцами.")
    print(f"Используем SMOTE с k_neighbors={k}")
    smote_multi = SMOTE(random_state=42, k_neighbors=k)
else:
    smote_multi = SMOTE(random_state=42)

try:
    X_train_multi_res, y_train_multi_res = smote_multi.fit_resample(X_train, y_train_multi)
    print(f"Размер мультиклассовой обучающей выборки после SMOTE: {X_train_multi_res.shape[0]}")
except ValueError as e:
    print(f"Предупреждение: SMOTE не применим для мультиклассовой задачи ({e})")
    print("Используем оригинальные данные без балансировки")
    X_train_multi_res, y_train_multi_res = X_train, y_train_multi

# ============================================================================
# 6. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================================
# --- Бинарная классификация ---
print("\n=== БИНАРНАЯ КЛАССИФИКАЦИЯ ===")

# Random Forest
print("Обучаем Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Attack']))

# XGBoost
print("Обучаем XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Benign', 'Attack']))

# ============================================================================
# КРОСС-ВАЛИДАЦИЯ
# ============================================================================
print("\n" + "="*50)
print("КРОСС-ВАЛИДАЦИЯ")
print("="*50)

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Используем стратифицированную кросс-валидацию
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Для XGBoost
print("\n5-кратная кросс-валидация (XGBoost):")
cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='f1')
print(f"F1-scores по фолдам: {cv_scores}")
print(f"Средний F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Для Random Forest
print("\n5-кратная кросс-валидация (Random Forest):")
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1')
print(f"F1-scores по фолдам: {cv_scores_rf}")
print(f"Средний F1-score: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Анализ результатов
if cv_scores.mean() < 0.99:
    print("\n Внимание: результаты кросс-валидации ниже, чем на тестовой выборке.")
    print("   Возможно, имеет место небольшое переобучение.")
else:
    print("\nМодель стабильна и показывает высокие результаты на всех фолдах.")

# --- Многоклассовая классификация (по типам атак) ---
print("\n=== МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ ===")
# Имена классов для отчета
class_names = label_encoder.classes_


# Получаем уникальные классы, которые реально присутствуют в тестовой выборке
unique_classes_in_test = np.unique(y_test_multi)
# Фильтруем имена классов и метки только для присутствующих классов
present_class_names = [class_names[i] for i in unique_classes_in_test]

# Random Forest
print("Обучаем Random Forest для многоклассовой задачи...")
rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_multi.fit(X_train_multi_res, y_train_multi_res)
y_pred_rf_multi = rf_multi.predict(X_test)
print("\nRandom Forest (Multiclass) Report:")
print(classification_report(y_test_multi, y_pred_rf_multi,
                           labels=unique_classes_in_test,
                           target_names=present_class_names))

# XGBoost
print("Обучаем XGBoost для многоклассовой задачи...")
xgb_multi = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb_multi.fit(X_train_multi_res, y_train_multi_res)
y_pred_xgb_multi = xgb_multi.predict(X_test)
print("\nXGBoost (Multiclass) Report:")
print(classification_report(y_test_multi, y_pred_xgb_multi,
                           labels=unique_classes_in_test,
                           target_names=present_class_names))
# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================================
# Выбираем лучшую модель для бинарной классификации (сравним по f1-score на классе Attack)
# Для простоты используем XGBoost, но можно реализовать выбор
best_model = xgb  # XGBoost обычно дает немного лучшие результаты

# Матрица ошибок для бинарной классификации
cm_binary = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6,5))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.title('Confusion Matrix (Binary Classification)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_binary.png')
plt.show()

# Матрица ошибок для многоклассовой классификации (XGBoost)
cm_multi = confusion_matrix(y_test_multi, y_pred_xgb_multi, labels=unique_classes_in_test)
plt.figure(figsize=(12,10))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Reds',
            xticklabels=present_class_names,
            yticklabels=present_class_names)
plt.title('Confusion Matrix (Multiclass Classification)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass.png')
plt.show()

# ============================================================================
# 8. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================================
# Получаем важность признаков из лучшей модели (бинарной)
feature_importance = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10,8))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Top 20 Feature Importance (XGBoost, Binary Classification)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nСкрипт завершен. Все графики сохранены в текущую директорию.")