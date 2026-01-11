 # ============================================
# 1. IMPORTAÇÕES PARA MODELAGEM
# ============================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurações visuais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# ============================================
# 2. PRÉ-PROCESSAMENTO DOS DADOS
# ============================================

print("="*60)
print("2. PRÉ-PROCESSAMENTO DOS DADOS")
print("="*60)

# 2.1. Carregar dados processados da EDA
df = pd.read_csv('diabetes_processed.csv')

# 2.2. Converter variável Age (de ordinal para contínuo)
def convert_age_to_numeric(age_code):
    """Converte código de faixa etária para valor numérico contínuo"""
    age_mapping = {
        1: 21,   # 18-24 → 21
        2: 27,   # 25-29 → 27
        3: 32,   # 30-34 → 32
        4: 37,   # 35-39 → 37
        5: 42,   # 40-44 → 42
        6: 47,   # 45-49 → 47
        7: 52,   # 50-54 → 52
        8: 57,   # 55-59 → 57
        9: 62,   # 60-64 → 62
        10: 67,  # 65-69 → 67
        11: 72,  # 70-74 → 72
        12: 77,  # 75-79 → 77
        13: 85   # 80+ → 85
    }
    return age_mapping.get(age_code, age_code)

# Aplicar conversão
df['Age_numeric'] = df['Age'].apply(convert_age_to_numeric)
print(f"\n2.1. Conversão de Age: Exemplo {df['Age'].iloc[0]} → {df['Age_numeric'].iloc[0]}")

# 2.3. Selecionar features para diferentes análises
print("\n2.2. Seleção de features:")

# Para clusterização
cluster_features = ['BMI', 'Age_numeric', 'GenHlth', 'PhysHlth', 
                    'MentHlth', 'HighBP', 'HighChol']
X_cluster = df[cluster_features]

# Para classificação (todas as features relevantes)
classification_features = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
    'PhysActivity', 'HvyAlcoholConsump', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'HeartDiseaseorAttack',
    'Age_numeric', 'Education', 'Income'
]

# Variável alvo
y = df['Diabetes_binary']

print(f"Features para clusterização: {len(cluster_features)} variáveis")
print(f"Features para classificação: {len(classification_features)} variáveis")

# 2.4. Normalização para clusterização
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

print(f"\n2.3. Normalização para clusterização concluída.")
print(f"  Shape: {X_cluster_scaled.shape}")
print(f"  Média após scaling: {X_cluster_scaled.mean():.2f}")
print(f"  Desvio padrão após scaling: {X_cluster_scaled.std():.2f}")

# ============================================
# 3. CLUSTERIZAÇÃO (K-MEANS)
# ============================================

print("\n" + "="*60)
print("3. CLUSTERIZAÇÃO COM K-MEANS")
print("="*60)

# 3.1. Método do cotovelo para determinar k ótimo
print("\n3.1. Calculando método do cotovelo...")
wcss = []
silhouette_scores = []
k_range = range(2, 21)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    wcss.append(kmeans.inertia_)
    
    # Calcular silhouette score (amostra para eficiência)
    from sklearn.metrics import silhouette_score
    if len(X_cluster_scaled) > 10000:
        sample_idx = np.random.choice(len(X_cluster_scaled), 10000, replace=False)
        silhouette_scores.append(silhouette_score(X_cluster_scaled[sample_idx], 
                                                  kmeans.labels_[sample_idx]))
    else:
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
    
    print(f"  k={k:2d} - WCSS: {wcss[-1]:.2f}, Silhouette: {silhouette_scores[-1]:.4f}")

# 3.2. Visualizar resultados
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico do cotovelo
axes[0].plot(k_range, wcss, marker='o', linestyle='--')
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)')
axes[0].set_title('Método do Cotovelo')
axes[0].grid(True)

# Gráfico do silhouette score
axes[1].plot(k_range, silhouette_scores, marker='o', linestyle='--', color='orange')
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por k')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3. Determinar k ótimo (baseado na análise)
# Normalmente escolhemos k no "cotovelo" da curva WCSS
# Vamos escolher k=4 para este exemplo
optimal_k = 4
print(f"\n3.2. k ótimo selecionado: {optimal_k}")
print(f"  Silhouette score para k={optimal_k}: {silhouette_scores[optimal_k-2]:.4f}")

# 3.4. Aplicar K-Means com k ótimo
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', 
                      random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_cluster_scaled)

# Adicionar clusters ao dataframe
df['Cluster'] = clusters

print(f"\n3.3. Clusterização concluída.")
print(f"  Tamanho dos clusters:")
for i in range(optimal_k):
    cluster_size = (df['Cluster'] == i).sum()
    perc = cluster_size / len(df) * 100
    print(f"    Cluster {i}: {cluster_size:,} observações ({perc:.1f}%)")

# 3.5. Análise dos clusters
print("\n3.4. Características médias por cluster:")
cluster_stats = df.groupby('Cluster')[cluster_features].mean()
print(cluster_stats)

# 3.6. Visualização dos clusters (usando PCA para 2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                      cmap='viridis', alpha=0.6, s=10)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title(f'Visualização dos Clusters (k={optimal_k})')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

# Adicionar centróides
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           marker='X', s=200, c='red', label='Centróides')
plt.legend()

plt.tight_layout()
plt.savefig('clusters_pca.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 4. PREPARAÇÃO PARA CLASSIFICAÇÃO
# ============================================

print("\n" + "="*60)
print("4. PREPARAÇÃO PARA CLASSIFICAÇÃO SUPERVISIONADA")
print("="*60)

# 4.1. Preparar dados para classificação
X_class = df[classification_features]

print(f"\n4.1. Dados para classificação:")
print(f"  X shape: {X_class.shape}")
print(f"  y shape: {y.shape}")

# 4.2. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n4.2. Divisão treino-teste:")
print(f"  Treino: {X_train.shape[0]:,} observações")
print(f"  Teste: {X_test.shape[0]:,} observações")
print(f"  Proporção diabetes (treino): {y_train.mean():.3f}")
print(f"  Proporção diabetes (teste): {y_test.mean():.3f}")

# 4.3. Balanceamento das classes (SMOTE apenas no treino)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\n4.3. Balanceamento com SMOTE:")
print(f"  Antes do SMOTE: {X_train.shape[0]:,} observações")
print(f"  Depois do SMOTE: {X_train_bal.shape[0]:,} observações")
print(f"  Proporção após SMOTE: {y_train_bal.mean():.3f}")
# ============================================
# 5. MODELAGEM: REGRESSÃO LOGÍSTICA OTIMIZADA
# ============================================

print("\n" + "="*60)
print("5. REGRESSÃO LOGÍSTICA - OTIMIZAÇÃO")
print("="*60)

# 5.1. OTIMIZAÇÃO DE HIPERPARÂMETROS PARA REGRESSÃO LOGÍSTICA
print("\n5.1. Otimizando hiperparâmetros da Regressão Logística...")

# Definir grade de parâmetros para otimização
param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga'],  # Ambos suportam L1 e L2
    'classifier__class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
    'classifier__max_iter': [1000, 2000]
}

# Criar pipeline base
pipeline_lr_base = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Otimizar com RandomizedSearchCV (mais rápido que GridSearchCV)
random_search_lr = RandomizedSearchCV(
    pipeline_lr_base,
    param_grid_lr,
    n_iter=50,  # Testar 50 combinações aleatórias
    cv=3,  # 3-fold cross-validation
    scoring='f1',  # Otimizar para F1-Score
    n_jobs=-1,  # Usar todos os núcleos
    random_state=42,
    verbose=1
)

print("Executando RandomizedSearchCV para Regressão Logística...")
random_search_lr.fit(X_train, y_train)  # Usar dados ORIGINAIS (não balanceados com SMOTE)

# 5.2. MELHORES PARÂMETROS ENCONTRADOS
print(f"\n5.2. Melhores parâmetros encontrados:")
for param, value in random_search_lr.best_params_.items():
    print(f"  {param}: {value}")

print(f"  Melhor F1-Score (validação): {random_search_lr.best_score_:.4f}")

# 5.3. TREINAR MODELO FINAL COM MELHORES PARÂMETROS
pipeline_lr_optimized = random_search_lr.best_estimator_

# Fazer previsões
y_pred_lr_opt = pipeline_lr_optimized.predict(X_test)
y_pred_proba_lr_opt = pipeline_lr_optimized.predict_proba(X_test)[:, 1]

# 5.4. AVALIAR MODELO OTIMIZADO
print("\n5.3. Métricas da Regressão Logística Otimizada:")
print("  Acurácia:", accuracy_score(y_test, y_pred_lr_opt))
print("  Precisão:", precision_score(y_test, y_pred_lr_opt))
print("  Recall:", recall_score(y_test, y_pred_lr_opt))
print("  F1-Score:", f1_score(y_test, y_pred_lr_opt))
print("  AUC-ROC:", roc_auc_score(y_test, y_pred_proba_lr_opt))

# 5.5. TESTAR DIFERENTES THRESHOLDS (ponto de corte)
print("\n5.4. Otimizando threshold (ponto de corte)...")

# Calcular curva ROC para encontrar melhor threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr_opt)

# Encontrar threshold que maximiza F1-Score
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_pred_proba_lr_opt >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"  Melhor threshold encontrado: {best_threshold:.2f}")
print(f"  F1-Score com threshold {best_threshold:.2f}: {best_f1:.4f}")

# Aplicar melhor threshold
y_pred_lr_best = (y_pred_proba_lr_opt >= best_threshold).astype(int)

print("\n5.5. Métricas com threshold otimizado:")
print("  Acurácia:", accuracy_score(y_test, y_pred_lr_best))
print("  Precisão:", precision_score(y_test, y_pred_lr_best))
print("  Recall:", recall_score(y_test, y_pred_lr_best))
print("  F1-Score:", f1_score(y_test, y_pred_lr_best))

# 5.6. VALIDAÇÃO CRUZADA DO MODELO OTIMIZADO
print(f"\n5.6. Validação Cruzada do modelo otimizado...")
cv_scores_lr_opt = cross_val_score(
    pipeline_lr_optimized, 
    X_train, 
    y_train, 
    cv=5, 
    scoring='f1',
    n_jobs=-1
)
print(f"  Scores F1: {cv_scores_lr_opt}")
print(f"  Média F1: {cv_scores_lr_opt.mean():.4f} (+/- {cv_scores_lr_opt.std()*2:.4f})")

# 5.7. ANÁLISE DETALHADA DOS RESULTADOS
print("\n" + "="*60)
print("ANÁLISE DETALHADA DOS RESULTADOS")
print("="*60)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_lr_best)
tn, fp, fn, tp = cm.ravel()

print(f"\nMatriz de Confusão:")
print(f"  Verdadeiros Negativos (TN): {tn}")
print(f"  Falsos Positivos (FP): {fp}")
print(f"  Falsos Negativos (FN): {fn}")
print(f"  Verdadeiros Positivos (TP): {tp}")

# Métricas calculadas manualmente
print(f"\nTaxas importantes:")
print(f"  Taxa de Falsos Positivos (FPR): {fp/(fp+tn):.4f}")
print(f"  Taxa de Falsos Negativos (FNR): {fn/(fn+tp):.4f}")
print(f"  Valor Preditivo Positivo (Precisão): {tp/(tp+fp):.4f}")
print(f"  Valor Preditivo Negativo: {tn/(tn+fn):.4f}")

# 5.8. COMPARAÇÃO COM OUTRAS TÉCNICAS DE BALANCEAMENTO
print("\n5.7. Testando diferentes técnicas de balanceamento...")

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Técnica 1: Under-sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# Técnica 2: SMOTE + Tomek
smote_tomek = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smote_tomek.fit_resample(X_train, y_train)

# Criar modelos para cada técnica
balance_methods = {
    'SMOTE (original)': (X_train_bal, y_train_bal),
    'Under-sampling': (X_train_rus, y_train_rus),
    'SMOTE + Tomek': (X_train_smt, y_train_smt),
    'Sem balanceamento': (X_train, y_train)
}

results_balance = []

for method_name, (X_train_m, y_train_m) in balance_methods.items():
    # Treinar modelo simples
    pipeline_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ))
    ])
    
    pipeline_temp.fit(X_train_m, y_train_m)
    y_pred_temp = pipeline_temp.predict(X_test)
    
    results_balance.append({
        'Método': method_name,
        'Amostras Treino': len(X_train_m),
        'Acurácia': accuracy_score(y_test, y_pred_temp),
        'F1-Score': f1_score(y_test, y_pred_temp),
        'Recall': recall_score(y_test, y_pred_temp),
        'Precisão': precision_score(y_test, y_pred_temp)
    })

# Mostrar resultados
results_balance_df = pd.DataFrame(results_balance)
print("\nComparação de técnicas de balanceamento:")
print(results_balance_df.to_string(index=False))

# 5.9. ESCOLHER MELHOR MODELO PARA REGRESSÃO LOGÍSTICA
print("\n" + "="*60)
print("DEFININDO MODELO FINAL DE REGRESSÃO LOGÍSTICA")
print("="*60)

# Usar o modelo otimizado como final
pipeline_lr = pipeline_lr_optimized
y_pred_lr = y_pred_lr_opt  # Ou y_pred_lr_best se quiser usar threshold otimizado
y_pred_proba_lr = y_pred_proba_lr_opt

print("Modelo final selecionado:")
print(f"  Tipo: Regressão Logística Otimizada")
print(f"  Parâmetros: {random_search_lr.best_params_}")
print(f"  F1-Score no teste: {f1_score(y_test, y_pred_lr):.4f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# 5.10. ANÁLISE DAS FEATURES IMPORTANTES
print("\n5.8. Análise das features mais importantes:")

# Extrair coeficientes do modelo
classifier = pipeline_lr.named_steps['classifier']
coefficients = classifier.coef_[0]
feature_names = X_train.columns

# Criar DataFrame com importância das features
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coeficiente': coefficients,
    'Importância Absoluta': np.abs(coefficients)
}).sort_values('Importância Absoluta', ascending=False)

print("\nTop 10 features mais importantes:")
print(feature_importance_df.head(10).to_string(index=False))

# Visualizar importância das features
plt.figure(figsize=(10, 6))
top_n = min(15, len(feature_importance_df))
top_features = feature_importance_df.head(top_n)

# Plot horizontal bars
bars = plt.barh(range(top_n), top_features['Importância Absoluta'][::-1])
plt.yticks(range(top_n), top_features['Feature'][::-1])
plt.xlabel('Importância Absoluta (|Coeficiente|)')
plt.title('Top 15 Features Mais Importantes - Regressão Logística')
plt.grid(axis='x', alpha=0.3)

# Adicionar valores nas barras
for i, (bar, coef) in enumerate(zip(bars, top_features['Coeficiente'][::-1])):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{coef:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance_logistic.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "-"*60)
print("REGRESSÃO LOGÍSTICA OTIMIZADA - CONCLUÍDA!")
print("-"*60)

# ============================================
# CONTINUA COM O RESTO DO CÓDIGO...
# NOTA: Atualizar as seções seguintes para usar y_pred_lr e y_pred_proba_lr
# ============================================
# ============================================
# 6. MODELAGEM: XGBOOST
# ============================================

print("\n" + "="*60)
print("6. XGBOOST")
print("="*60)

# 6.1. Criar modelo XGBoost
xgb_model = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train_bal[y_train_bal==0]) / len(y_train_bal[y_train_bal==1]),
    eval_metric='logloss',
    use_label_encoder=False
)

# 6.2. Treinar modelo
xgb_model.fit(X_train_bal, y_train_bal)

# 6.3. Fazer previsões
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# 6.4. Avaliar modelo
print("\n6.1. Métricas do XGBoost:")
print("  Acurácia:", accuracy_score(y_test, y_pred_xgb))
print("  Precisão:", precision_score(y_test, y_pred_xgb))
print("  Recall:", recall_score(y_test, y_pred_xgb))
print("  F1-Score:", f1_score(y_test, y_pred_xgb))
print("  AUC-ROC:", roc_auc_score(y_test, y_pred_proba_xgb))

# 6.5. Validação cruzada
cv_scores_xgb = cross_val_score(xgb_model, X_train_bal, y_train_bal, 
                                cv=5, scoring='f1')
print(f"\n6.2. Validação Cruzada (F1-Score):")
print(f"  Scores: {cv_scores_xgb}")
print(f"  Média: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std()*2:.4f})")

# ============================================
# 7. OTIMIZAÇÃO DE HIPERPARÂMETROS
# ============================================

print("\n" + "="*60)
print("7. OTIMIZAÇÃO DE HIPERPARÂMETROS (XGBOOST)")
print("="*60)

# 7.1. Definir grade de parâmetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# 7.2. Busca aleatória (mais eficiente para grandes grids)
xgb_opt = XGBClassifier(random_state=42, use_label_encoder=False)

random_search = RandomizedSearchCV(
    xgb_opt, param_grid, n_iter=20, 
    scoring='f1', cv=3, random_state=42, n_jobs=-1
)

print("7.1. Executando Randomized Search...")
random_search.fit(X_train_bal, y_train_bal)

# 7.3. Melhores parâmetros
print(f"\n7.2. Melhores parâmetros encontrados:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"  Melhor F1-Score (validação): {random_search.best_score_:.4f}")

# 7.4. Treinar modelo otimizado
xgb_optimized = random_search.best_estimator_

# Avaliar no teste
y_pred_opt = xgb_optimized.predict(X_test)
y_pred_proba_opt = xgb_optimized.predict_proba(X_test)[:, 1]

print(f"\n7.3. Métricas do XGBoost Otimizado:")
print("  Acurácia:", accuracy_score(y_test, y_pred_opt))
print("  F1-Score:", f1_score(y_test, y_pred_opt))
print("  AUC-ROC:", roc_auc_score(y_test, y_pred_proba_opt))
# ============================================
# 8. ANÁLISE COMPARATIVA DOS MODELOS
# ============================================

print("\n" + "="*60)
print("8. COMPARAÇÃO DOS MODELOS")
print("="*60)

# 8.1. Criar tabela comparativa - USAR AS VARIÁVEIS ATUALIZADAS
comparison_data = {
    'Modelo': ['Regressão Logística (Otimizada)', 'XGBoost (Padrão)', 'XGBoost (Otimizado)'],  # Mudar nome
    'Acurácia': [
        accuracy_score(y_test, y_pred_lr),  # Já atualizado na seção 5
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_opt)
    ],
    'Precisão': [
        precision_score(y_test, y_pred_lr),  # Já atualizado
        precision_score(y_test, y_pred_xgb),
        precision_score(y_test, y_pred_opt)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),  # Já atualizado
        recall_score(y_test, y_pred_xgb),
        recall_score(y_test, y_pred_opt)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),  # Já atualizado
        f1_score(y_test, y_pred_xgb),
        f1_score(y_test, y_pred_opt)
    ],
    'AUC-ROC': [
        roc_auc_score(y_test, y_pred_proba_lr),  # Já atualizado
        roc_auc_score(y_test, y_pred_proba_xgb),
        roc_auc_score(y_test, y_pred_proba_opt)
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n8.1. Comparação de Desempenho:")
print(comparison_df.to_string(index=False))

# 8.2. Visualização comparativa (já está usando as variáveis corretas)

# 8.3. Matriz de confusão comparativa - ATUALIZAR
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [
    ('Regressão Logística (Otimizada)', y_pred_lr),  # Atualizar nome e variável
    ('XGBoost (Padrão)', y_pred_xgb),
    ('XGBoost (Otimizado)', y_pred_opt)
]

for i, (name, y_pred) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Matriz de Confusão - {name}')
    axes[i].set_xlabel('Predito')
    axes[i].set_ylabel('Real')
    axes[i].set_xticklabels(['Não', 'Sim'])
    axes[i].set_yticklabels(['Não', 'Sim'])

plt.tight_layout()
plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

# 8.4. Curvas ROC comparativas - ATUALIZAR
plt.figure(figsize=(10, 8))

# Calcular curvas ROC - USAR VARIÁVEIS ATUALIZADAS
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)  # Já atualizado
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_pred_proba_opt)

# Plotar curvas - ATUALIZAR NOME
plt.plot(fpr_lr, tpr_lr, label=f'Regressão Logística Otimizada (AUC = {roc_auc_score(y_test, y_pred_proba_lr):.3f})', linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost Padrão (AUC = {roc_auc_score(y_test, y_pred_proba_xgb):.3f})', linewidth=2)
plt.plot(fpr_opt, tpr_opt, label=f'XGBoost Otimizado (AUC = {roc_auc_score(y_test, y_pred_proba_opt):.3f})', linewidth=2)

# Linha de referência (aleatório)
plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório')

plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC Comparativas')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 9. EXPLICABILIDADE COM SHAP
# ============================================

print("\n" + "="*60)
print("9. EXPLICABILIDADE DO MODELO (SHAP)")
print("="*60)

# 9.1. Explicar modelo XGBoost otimizado
print("\n9.1. Calculando valores SHAP...")

# Amostrar para eficiência
sample_idx = np.random.choice(X_test.shape[0], min(1000, X_test.shape[0]), replace=False)
X_test_sample = X_test.iloc[sample_idx]

# Criar explainer SHAP
explainer = shap.TreeExplainer(xgb_optimized)
shap_values = explainer.shap_values(X_test_sample)

# 9.2. Gráfico de importância global
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title('Importância Global das Features (SHAP)')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.3. Gráfico de resumo (beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('Impacto das Features nas Previsões (SHAP)')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.4. Explicação individual
print("\n9.2. Exemplo de explicação individual:")
# Escolher uma observação específica
idx_example = 0  # Primeira observação da amostra

shap.force_plot(explainer.expected_value, shap_values[idx_example,:], 
                X_test_sample.iloc[idx_example,:], matplotlib=True, show=False)
plt.title(f'Explicação Individual - Observação {idx_example}')
plt.tight_layout()
plt.savefig('shap_individual.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.5. Valores SHAP médios por feature
shap_df = pd.DataFrame({
    'feature': X_test_sample.columns,
    'shap_abs_mean': np.abs(shap_values).mean(axis=0),
    'shap_mean': shap_values.mean(axis=0)
}).sort_values('shap_abs_mean', ascending=False)

print("\n9.3. Importância média das features (SHAP):")
print(shap_df.head(10).to_string(index=False))
# ============================================
# 10. SALVAR MODELOS E RESULTADOS
# ============================================

print("\n" + "="*60)
print("10. SALVANDO MODELOS E RESULTADOS")
print("="*60)

import joblib
import json

# 10.1. Salvar modelos - ATUALIZAR PARA pipeline_lr (não lr_model)
joblib.dump(pipeline_lr, 'modelo_regressao_logistica_otimizada.pkl')  # MUDAR NOME
joblib.dump(xgb_optimized, 'modelo_xgboost_otimizado.pkl')
# Remover scalers desnecessários
joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
joblib.dump(kmeans_final, 'modelo_kmeans.pkl')

print("10.1. Modelos salvos:")
print("  - modelo_regressao_logistica_otimizada.pkl")  # Atualizado
print("  - modelo_xgboost_otimizado.pkl")
print("  - scaler_cluster.pkl")
print("  - modelo_kmeans.pkl")

# 10.2. Salvar resultados da comparação (já está atualizado)

# 10.3. Salvar configuração final - ATUALIZAR
config = {
    'cluster_features': cluster_features,
    'classification_features': classification_features,
    'optimal_k': optimal_k,
    'best_model': 'XGBoost (Otimizado)',  # Pode mudar após otimização
    'best_f1_score': float(f1_score(y_test, y_pred_opt)),  # Atualizar se necessário
    'best_auc_roc': float(roc_auc_score(y_test, y_pred_proba_opt)),  # Atualizar se necessário
    'lr_f1_score': float(f1_score(y_test, y_pred_lr)),  # Adicionar métricas da LR
    'lr_auc_roc': float(roc_auc_score(y_test, y_pred_proba_lr)),  # Adicionar métricas da LR
    'top_features_shap': shap_df['feature'].head(5).tolist()
}

# Verificar qual modelo é melhor
if f1_score(y_test, y_pred_lr) > f1_score(y_test, y_pred_opt):
    config['best_model'] = 'Regressão Logística (Otimizada)'
    config['best_f1_score'] = float(f1_score(y_test, y_pred_lr))
    config['best_auc_roc'] = float(roc_auc_score(y_test, y_pred_proba_lr))

with open('configuracao_modelo.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\n10.3. Configuração salva em 'configuracao_modelo.json'")

# 10.5. Relatório final - ATUALIZAR
print("\n" + "="*60)
print("RELATÓRIO FINAL")
print("="*60)
print(f"1. Dataset: {len(df):,} observações, {len(df.columns)} variáveis")
print(f"2. Clusterização: {optimal_k} clusters identificados")
print(f"3. Modelo vencedor: {config['best_model']}")
print(f"4. Desempenho (F1-Score): {config['best_f1_score']:.4f}")
print(f"5. Desempenho (AUC-ROC): {config['best_auc_roc']:.4f}")
print(f"6. Features mais importantes: {', '.join(config['top_features_shap'])}")

# Adicionar comparação entre modelos
print(f"\nComparação detalhada:")
print(f"  Regressão Logística - F1: {f1_score(y_test, y_pred_lr):.4f}, AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
print(f"  XGBoost Otimizado - F1: {f1_score(y_test, y_pred_opt):.4f}, AUC: {roc_auc_score(y_test, y_pred_proba_opt):.4f}")

print("\n" + "="*60)
print("METODOLOGIA CONCLUÍDA!")
print("="*60)