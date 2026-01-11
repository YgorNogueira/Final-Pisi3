import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)
import joblib
import json
import lightgbm as lgb

print("1. CARREGAMENTO DOS DADOS")
csv_path = "diabetes_processed.csv"

df = pd.read_csv(csv_path)
print(f"Dados carregados: {df.shape[0]} observações, {df.shape[1]} variáveis")

target_col = "Diabetes_binary"
if target_col not in df.columns:
    raise SystemExit(1)

print("2. PREPARAÇÃO DOS DADOS")

classification_features = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "PhysActivity", "HvyAlcoholConsump", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "HeartDiseaseorAttack",
    "Age", "Education", "Income"
]

available_features = [f for f in classification_features if f in df.columns]
missing_features = [f for f in classification_features if f not in df.columns]

print(f"Usando {len(available_features)} features: {available_features}")

if len(available_features) < 5:
    raise SystemExit(1)

X = df[available_features].copy()
y = df[target_col].astype(int).copy()

print("\nEstatísticas do dataset:")
print(f"  Total de observações: {len(df):,}")
print(f"  Features utilizadas: {len(available_features)}")
print(f"  Casos de diabetes (classe 1): {y.sum():,} ({y.mean():.2%})")


print("3. DIVISÃO DOS DADOS (Treino / Validação / Teste)")

# 80/20 (treino+val / teste)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# dentro do 80%, separar validação (ex: 20% do trainval => 64/16/20 no total)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
)

print("Tamanhos:")
print(f"  Treino:     {X_train.shape[0]:,}")
print(f"  Validação:  {X_val.shape[0]:,}")
print(f"  Teste:      {X_test.shape[0]:,}")
print("Proporções classe positiva:")
print(f"  Treino:     {y_train.mean():.3f} ({y_train.sum():,} casos)")
print(f"  Validação:  {y_val.mean():.3f} ({y_val.sum():,} casos)")
print(f"  Teste:      {y_test.mean():.3f} ({y_test.sum():,} casos)")


print("4. DESBALANCEAMENTO: scale_pos_weight (sem SMOTE)")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
if pos == 0:
    raise SystemExit("Não há exemplos positivos no treino. Verifique o dataset.")

scale_pos_weight = neg / pos
print(f"  Negativos (0): {neg:,}")
print(f"  Positivos (1): {pos:,}")
print(f"  scale_pos_weight = {scale_pos_weight:.4f}")

print("5. TREINAMENTO DO LIGHTGBM (early stopping)")

lgb_model = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.03,
    n_estimators=4000,           
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=True)]
)
print(f"Melhor iteração (best_iteration_): {getattr(lgb_model, 'best_iteration_', None)}")


print("6. THRESHOLD OTIMIZADO")

# Probabilidades na validação para escolher threshold
val_proba = lgb_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
f1_scores_thr = f1_scores[1:]
precision_thr = precision[1:]
recall_thr = recall[1:]

best_idx = int(np.argmax(f1_scores_thr))
best_threshold = float(thresholds[best_idx])

print(f"Melhor threshold: {best_threshold:.4f}")
print(f"F1 (val):        {f1_scores_thr[best_idx]:.4f}")
print(f"Precisão (val):  {precision_thr[best_idx]:.4f}")
print(f"Recall (val):    {recall_thr[best_idx]:.4f}")

print("7. AVALIAÇÃO NO TESTE (threshold ajustado)")

y_test_proba = lgb_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
auc = roc_auc_score(y_test, y_test_proba)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print("Métricas de desempenho (TESTE):")
print("=" * 50)
print(f"{'Threshold:':<15} {best_threshold:.4f}")
print(f"{'Acurácia:':<15} {acc:.4f}")
print(f"{'Precisão:':<15} {prec:.4f}")
print(f"{'Recall:':<15} {rec:.4f}")
print(f"{'F1-Score:':<15} {f1:.4f}")
print(f"{'AUC-ROC:':<15} {auc:.4f}")
print("=" * 50)

print("\nMatriz de Confusão (TESTE):")
print("=" * 50)
print(f"                Predito")
print(f"              Negativo Positivo")
print(f"Real Negativo {tn:>8} {fp:>8}")
print(f"     Positivo {fn:>8} {tp:>8}")
print("=" * 50)

print("\nRelatório de classificação (TESTE):")
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

# Taxas derivadas
print("\nTaxas importantes (TESTE):")
tpr = tp / (tp + fn + 1e-12)  
tnr = tn / (tn + fp + 1e-12)  
fpr = fp / (fp + tn + 1e-12)
fnr = fn / (fn + tp + 1e-12)
print(f"  Sensibilidade (TPR/Recall): {tpr:.4f}")
print(f"  Especificidade (TNR):       {tnr:.4f}")
print(f"  Taxa de falsos positivos:   {fpr:.4f}")
print(f"  Taxa de falsos negativos:   {fnr:.4f}")

print("8. CURVA ROC")
fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr_curve, tpr_curve, lw=2, label=f"LightGBM (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Aleatório (AUC = 0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - LightGBM Melhorado")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lightgbm_melhorado_curva_roc.png", dpi=300, bbox_inches="tight")


print("9. IMPORTÂNCIA DAS FEATURES")

feature_importance = lgb_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importância": feature_importance
}).sort_values("Importância", ascending=False)

print("\nTop 10 features mais importantes:")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(12, 8))
top_n = min(15, len(importance_df))
top_features = importance_df.head(top_n)

plt.barh(range(top_n), top_features["Importância"][::-1])
plt.yticks(range(top_n), top_features["Feature"][::-1])
plt.xlabel("Importância")
plt.title("Top 15 Features Mais Importantes - LightGBM Melhorado")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("lightgbm_melhorado_feature_importance.png", dpi=300, bbox_inches="tight")

joblib.dump(lgb_model, "modelo_lightgbm_melhorado.pkl")

metrics_dict = {
    "threshold": float(best_threshold),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "auc_roc": float(auc),
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    },
    "top_features": importance_df.head(10).to_dict("records"),
    "scale_pos_weight": float(scale_pos_weight),
    "best_iteration": int(getattr(lgb_model, "best_iteration_", 0) or 0),
}

with open("lightgbm_melhorado_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

predictions_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_test_pred,
    "probability": y_test_proba
})
predictions_df.to_csv("lightgbm_melhorado_predictions.csv", index=False)


print("RELATÓRIO FINAL - LIGHTGBM MELHORADO")
print(f"""
RESUMO:

• Estratégias aplicadas:
  - Remoção do SMOTE 
  - Ajuste de desbalanceamento com scale_pos_weight = {scale_pos_weight:.4f}
  - Early stopping com validação
  - Otimização automática do threshold com base no F1 max na validação

• Desempenho no TESTE (threshold ajustado):
  - Threshold: {best_threshold:.4f}
  - Acurácia:  {acc:.4f}
  - Precisão:  {prec:.4f}
  - Recall:    {rec:.4f}
  - F1-Score:  {f1:.4f}
  - AUC-ROC:   {auc:.4f}
""")