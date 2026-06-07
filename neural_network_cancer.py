# ============================================================
#  Classificação de Câncer de Mama com Redes Neurais (Keras)
#  Dataset: wisc_bc_data.csv
#  Objetivo: Classificar tumor como Maligno (M) ou Benigno (B)
# ============================================================

# ── 1. IMPORTAÇÕES ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow versão:", tf.__version__)

# ── 2. CARREGAR DADOS ────────────────────────────────────────
url = ("https://raw.githubusercontent.com/stedy/"
       "Machine-Learning-with-R-datasets/master/wisc_bc_data.csv")

df = pd.read_csv(url)
print("\nShape:", df.shape)
print(df.head())

# ── 3. ANÁLISE EXPLORATÓRIA (EDA) ────────────────────────────
print("\n--- Informações gerais ---")
print(df.info())

print("\n--- Valores nulos ---")
print(df.isnull().sum())

print("\n--- Distribuição do target ---")
print(df['diagnosis'].value_counts())

# Gráfico 1: Distribuição do target
plt.figure(figsize=(5, 4))
df['diagnosis'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Distribuição: Benigno (B) vs Maligno (M)')
plt.xlabel('Diagnóstico')
plt.ylabel('Quantidade')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_distribuicao_target.png', dpi=150)
plt.show()
print("Gráfico salvo: eda_distribuicao_target.png")

# Gráfico 2: Correlação entre features
plt.figure(figsize=(14, 10))
numeric_df = df.drop(columns=['id', 'diagnosis'])
sns.heatmap(numeric_df.corr(), cmap='coolwarm', center=0, linewidths=0.3)
plt.title('Mapa de Correlação entre Features')
plt.tight_layout()
plt.savefig('eda_correlacao.png', dpi=150)
plt.show()
print("Gráfico salvo: eda_correlacao.png")

# Gráfico 3: Boxplot das principais features por diagnóstico
features_plot = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, feat in zip(axes, features_plot):
    df.boxplot(column=feat, by='diagnosis', ax=ax)
    ax.set_title(feat)
    ax.set_xlabel('Diagnóstico')
plt.suptitle('Distribuição das Features por Diagnóstico')
plt.tight_layout()
plt.savefig('eda_boxplot.png', dpi=150)
plt.show()
print("Gráfico salvo: eda_boxplot.png")

# ── 4. PRÉ-PROCESSAMENTO ─────────────────────────────────────
# Remover coluna ID (não é feature útil)
df = df.drop(columns=['id'])

# Separar features e target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Codificar target: B=0, M=1
le = LabelEncoder()
y = le.fit_transform(y)
print("\nClasses:", le.classes_, "→ [0, 1]")

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTreino: {X_train.shape[0]} amostras")
print(f"Teste:  {X_test.shape[0]} amostras")

# Normalizar features (média=0, desvio=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. CONSTRUÇÃO DA REDE NEURAL ─────────────────────────────
# Arquitetura: Input → 64 → Dropout → 32 → Dropout → 16 → Output
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),   # Camada oculta 1
    Dropout(0.3),
    Dense(16, activation='relu'),   # Camada oculta 2
    Dense(1,  activation='sigmoid') # Saída binária
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── 6. TREINAMENTO ───────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ── 7. CURVAS DE APRENDIZADO ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Treino')
ax1.plot(history.history['val_accuracy'], label='Validação')
ax1.set_title('Acurácia por Época')
ax1.set_xlabel('Época')
ax1.set_ylabel('Acurácia')
ax1.legend()

ax2.plot(history.history['loss'],     label='Treino')
ax2.plot(history.history['val_loss'], label='Validação')
ax2.set_title('Loss por Época')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('treinamento_curvas.png', dpi=150)
plt.show()
print("Gráfico salvo: treinamento_curvas.png")

# ── 8. AVALIAÇÃO NO CONJUNTO DE TESTE ────────────────────────
y_pred_prob = model.predict(X_test)
y_pred      = (y_pred_prob >= 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"  ACURÁCIA NO TESTE: {acc*100:.2f}%")
print(f"{'='*40}")
print("\nRelatório completo:")
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title(f'Matriz de Confusão  |  Acurácia: {acc*100:.2f}%')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=150)
plt.show()
print("Gráfico salvo: matriz_confusao.png")

print("\n Feito")
