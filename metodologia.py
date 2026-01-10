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
# 5. MODELAGEM: REGRESSÃO LOGÍSTICA
# ============================================

print("\n" + "="*60)
print("5. REGRESSÃO LOGÍSTICA")
print("="*60)

# 5.1. Normalizar features para Regressão Logística
scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train_bal)
X_test_lr = scaler_lr.transform(X_test)

# 5.2. Criar e treinar modelo
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_lr, y_train_bal)

# 5.3. Fazer previsões
y_pred_lr = lr_model.predict(X_test_lr)
y_pred_proba_lr = lr_model.predict_proba(X_test_lr)[:, 1]

# 5.4. Avaliar modelo
print("\n5.1. Métricas da Regressão Logística:")
print("  Acurácia:", accuracy_score(y_test, y_pred_lr))
print("  Precisão:", precision_score(y_test, y_pred_lr))
print("  Recall:", recall_score(y_test, y_pred_lr))
print("  F1-Score:", f1_score(y_test, y_pred_lr))
print("  AUC-ROC:", roc_auc_score(y_test, y_pred_proba_lr))

# 5.5. Validação cruzada
cv_scores_lr = cross_val_score(lr_model, X_train_lr, y_train_bal, 
                               cv=5, scoring='f1')
print(f"\n5.2. Validação Cruzada (F1-Score):")
print(f"  Scores: {cv_scores_lr}")
print(f"  Média: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std()*2:.4f})")

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

# 8.1. Criar tabela comparativa
comparison_data = {
    'Modelo': ['Regressão Logística', 'XGBoost (Padrão)', 'XGBoost (Otimizado)'],
    'Acurácia': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_opt)
    ],
    'Precisão': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_xgb),
        precision_score(y_test, y_pred_opt)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_xgb),
        recall_score(y_test, y_pred_opt)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_xgb),
        f1_score(y_test, y_pred_opt)
    ],
    'AUC-ROC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_xgb),
        roc_auc_score(y_test, y_pred_proba_opt)
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n8.1. Comparação de Desempenho:")
print(comparison_df.to_string(index=False))

# 8.2. Visualização comparativa
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    ax.bar(comparison_df['Modelo'], comparison_df[metric], color=['blue', 'orange', 'green'])
    ax.set_title(f'Comparação de {metric}')
    ax.set_ylabel(metric)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores
    for j, val in enumerate(comparison_df[metric]):
        ax.text(j, val + 0.01, f'{val:.3f}', ha='center', va='bottom')

# 8.3. Matriz de confusão comparativa
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [('Regressão Logística', y_pred_lr), 
          ('XGBoost (Padrão)', y_pred_xgb),
          ('XGBoost (Otimizado)', y_pred_opt)]

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

# 8.4. Curvas ROC comparativas
plt.figure(figsize=(10, 8))

# Calcular curvas ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_pred_proba_opt)

# Plotar curvas
plt.plot(fpr_lr, tpr_lr, label=f'Regressão Logística (AUC = {roc_auc_score(y_test, y_pred_proba_lr):.3f})', linewidth=2)
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

# 10.1. Salvar modelos
joblib.dump(lr_model, 'modelo_regressao_logistica.pkl')
joblib.dump(xgb_optimized, 'modelo_xgboost_otimizado.pkl')
joblib.dump(scaler_lr, 'scaler_regressao.pkl')
joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
joblib.dump(kmeans_final, 'modelo_kmeans.pkl')

print("10.1. Modelos salvos:")
print("  - modelo_regressao_logistica.pkl")
print("  - modelo_xgboost_otimizado.pkl")
print("  - scaler_regressao.pkl")
print("  - scaler_cluster.pkl")
print("  - modelo_kmeans.pkl")

# 10.2. Salvar resultados da comparação
comparison_df.to_csv('resultados_comparacao.csv', index=False)

# 10.3. Salvar resultados da clusterização
cluster_summary = df.groupby('Cluster').agg({
    'Diabetes_binary': ['mean', 'count'],
    'BMI': 'mean',
    'Age_numeric': 'mean',
    'GenHlth': 'mean',
    'HighBP': 'mean'
})
cluster_summary.columns = ['Taxa_Diabetes', 'n_observacoes', 'BMI_medio', 
                          'Idade_media', 'Saude_Geral_media', 'Hipertensao_media']
cluster_summary.to_csv('resumo_clusters.csv')

print("\n10.2. Resultados salvos:")
print("  - resultados_comparacao.csv")
print("  - resumo_clusters.csv")

# 10.4. Salvar configuração final
config = {
    'cluster_features': cluster_features,
    'classification_features': classification_features,
    'optimal_k': optimal_k,
    'best_model': 'XGBoost (Otimizado)',
    'best_f1_score': float(f1_score(y_test, y_pred_opt)),
    'best_auc_roc': float(roc_auc_score(y_test, y_pred_proba_opt)),
    'top_features_shap': shap_df['feature'].head(5).tolist()
}

with open('configuracao_modelo.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\n10.3. Configuração salva em 'configuracao_modelo.json'")

# 10.5. Relatório final
print("\n" + "="*60)
print("RELATÓRIO FINAL")
print("="*60)
print(f"1. Dataset: {len(df):,} observações, {len(df.columns)} variáveis")
print(f"2. Clusterização: {optimal_k} clusters identificados")
print(f"3. Modelo vencedor: {config['best_model']}")
print(f"4. Desempenho (F1-Score): {config['best_f1_score']:.4f}")
print(f"5. Desempenho (AUC-ROC): {config['best_auc_roc']:.4f}")
print(f"6. Features mais importantes: {', '.join(config['top_features_shap'])}")

print("\n" + "="*60)
print("METODOLOGIA CONCLUÍDA!")
print("="*60)