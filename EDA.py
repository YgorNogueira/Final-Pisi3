import pandas as pd
import io
import contextlib

import dash
from dash import html, dash_table

import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np


df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

def get_df_info(df):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()

def fig_to_base64():
    """Converte a figura atual do matplotlib em base64 (data URI)."""
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_boxplot_overall(df):
    """Boxplots gerais (como você já tinha) - bom para outliers."""
    plt.figure(figsize=(15, 15))
    cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age','Education', 'Income']
    for i, col in enumerate(cols):
        plt.subplot(4, 2, i + 1)
        sns.boxplot(x=col, data=df)
        plt.title(f"Boxplot - {col}")
    return fig_to_base64()

def generate_target_distribution(df, target="Diabetes_binary"):
    """Distribuição do target (balanceamento)."""
    counts = df[target].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title(f"Distribuição do alvo ({target})")
    plt.xlabel("Classe")
    plt.ylabel("Contagem")
    return fig_to_base64()

def generate_numeric_hist_by_target(df, col, target="Diabetes_binary", bins=30):
    """Histograma por classe (overlay) para variáveis numéricas/ordinais."""
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, hue=target, bins=bins, stat="count", common_norm=False, alpha=0.5)
    plt.title(f"Distribuição de {col} por {target}")
    plt.xlabel(col)
    plt.ylabel("Contagem")
    return fig_to_base64()

def generate_numeric_box_by_target(df, col, target="Diabetes_binary"):
    """Boxplot estratificado por target."""
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x=target, y=col)
    plt.title(f"{col} por {target}")
    plt.xlabel(target)
    plt.ylabel(col)
    return fig_to_base64()

def generate_binary_rate_plot(df, col, target="Diabetes_binary"):
    """
    Para variável binária (0/1): mostra taxa média de diabetes em cada categoria.
    Isso dá um insight tipo: "quem tem HighBP tem maior % de diabetes".
    """
    rate = df.groupby(col)[target].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=rate, x=col, y=target)
    plt.title(f"Taxa média de {target} por {col}")
    plt.xlabel(col)
    plt.ylabel(f"Média de {target} (proporção)")
    return fig_to_base64()

def generateHeatMap(df, method="spearman", show_annot=False):
    corr = df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(18, 12))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
        annot=show_annot,
        fmt=".2f"
    )
    plt.title(f"Correlation Heatmap ({method.title()})", pad=12)
    return fig_to_base64()

def generateTargetCorrelationPlot(df, target="Diabetes_binary", method="spearman"):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    correlationSeries = df.corr(method=method)[target].drop(target).sort_values()

    plt.figure(figsize=(10, max(6, len(correlationSeries) * 0.35)))
    sns.barplot(x=correlationSeries.values, y=correlationSeries.index)
    plt.title(f"Correlação de variáveis com o {target} ({method.title()})", pad=12)
    plt.xlabel("Coeficiente de Correlação")
    plt.ylabel("Variáveis")
    return fig_to_base64()

def build_group_stats_table(df, target="Diabetes_binary", cols=None):
    """
    Tabela de médias e medianas por classe do target.
    Isso é MUITO útil para o artigo (transforma EDA em análise).
    """
    if cols is None:
        cols = ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth', 'Income', 'Education']

    grouped_mean = df.groupby(target)[cols].mean().round(3)
    grouped_median = df.groupby(target)[cols].median().round(3)

    mean_df = grouped_mean.reset_index().rename(columns={target: "Classe"})
    mean_df.insert(1, "Métrica", "Média")

    median_df = grouped_median.reset_index().rename(columns={target: "Classe"})
    median_df.insert(1, "Métrica", "Mediana")

    out = pd.concat([mean_df, median_df], ignore_index=True)
    return out


# ===== Pré-processamento =====
db = df.copy()

# transformando tudo em int
for col in db.columns:
    db[col] = db[col].astype(int)

# removendo duplicatas
duplicates_before = db.duplicated().sum()
db.drop_duplicates(inplace=True)
duplicates_after = db.duplicated().sum()

shape_text = f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}"
info_text = get_df_info(df)

head_df = df.head(5)
describe_df = df.describe().reset_index()

shape_text2 = f"Linhas: {db.shape[0]} | Colunas: {db.shape[1]}"
info_text2 = get_df_info(db)

nulls_df = db.isnull().sum().reset_index()
nulls_df.columns = ["Coluna", "Valores nulos"]

unique_values = {col: db[col].nunique() for col in db.columns}
unique_df = pd.DataFrame.from_dict(unique_values, orient="index", columns=["Unique value count"]) \
    .reset_index().rename(columns={"index": "Coluna"})

target = "Diabetes_binary"

target_dist_img = generate_target_distribution(db, target=target)

boxplot_img = generate_boxplot_overall(db)

heatmap_img = generateHeatMap(db, method="spearman", show_annot=False)
target_corr_img = generateTargetCorrelationPlot(db, target=target, method="spearman")

hist_bmi_img = generate_numeric_hist_by_target(db, "BMI", target=target, bins=35)
hist_age_img = generate_numeric_hist_by_target(db, "Age", target=target, bins=13)
hist_genhlth_img = generate_numeric_hist_by_target(db, "GenHlth", target=target, bins=5)
hist_physhlth_img = generate_numeric_hist_by_target(db, "PhysHlth", target=target, bins=31)

box_bmi_target_img = generate_numeric_box_by_target(db, "BMI", target=target)
box_age_target_img = generate_numeric_box_by_target(db, "Age", target=target)
box_genhlth_target_img = generate_numeric_box_by_target(db, "GenHlth", target=target)
box_physhlth_target_img = generate_numeric_box_by_target(db, "PhysHlth", target=target)
box_menthlth_target_img = generate_numeric_box_by_target(db, "MentHlth", target=target)

rate_highbp_img = generate_binary_rate_plot(db, "HighBP", target=target)
rate_highchol_img = generate_binary_rate_plot(db, "HighChol", target=target)
rate_physactivity_img = generate_binary_rate_plot(db, "PhysActivity", target=target)

group_stats_df = build_group_stats_table(db, target=target)


# ===== Dash app =====
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={"padding": "20px"}, children=[

    html.H1("EDA – Diabetes Health Indicators Dataset"),
    html.H2("Link para o dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"),

    html.H2("1) Visão geral do Dataset"),
    html.H3("Shape do Dataset"),
    html.P(shape_text),
    html.H3("Informações do Dataset"),
    html.Pre(info_text, style={"backgroundColor": "#f4f4f4", "padding": "10px"}),

    html.H3("Primeiras 5 linhas"),
    dash_table.DataTable(
        data=head_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in head_df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    ),

    html.H3("Estatísticas Descritivas"),
    dash_table.DataTable(
        data=describe_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in describe_df.columns],
        style_table={"overflowX": "auto"}
    ),

    html.H2("2) Pré-processamento básico"),
    html.H3("Conversão para int + remoção de duplicatas"),
    html.P(shape_text2),
    html.Pre(info_text2, style={"backgroundColor": "#f4f4f4", "padding": "10px"}),

    html.H3("Valores Nulos"),
    dash_table.DataTable(
        data=nulls_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in nulls_df.columns],
        style_table={"overflowX": "auto"}
    ),

    html.H3("Valores Únicos por Atributo"),
    dash_table.DataTable(
        data=unique_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in unique_df.columns],
        style_table={"overflowX": "auto"}
    ),

    html.H3("Duplicatas"),
    html.P(f"Antes: {duplicates_before} | Depois: {duplicates_after}"),

    html.H2("3) Distribuição do alvo (balanceamento)"),
    html.Img(src=target_dist_img, style={"width": "60%"}),

    html.H2("4) Comparação por classe (Sem diabetes vs Com diabetes)"),
    html.H3("Tabela de médias e medianas por classe"),
    dash_table.DataTable(
        data=group_stats_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in group_stats_df.columns],
        style_table={"overflowX": "auto"},
        page_size=10
    ),

    html.H3("Histogramas por classe (principais)"),
    html.H4("BMI por Diabetes"),
    html.Img(src=hist_bmi_img, style={"width": "100%"}),
    html.H4("Age por Diabetes"),
    html.Img(src=hist_age_img, style={"width": "100%"}),
    html.H4("GenHlth por Diabetes"),
    html.Img(src=hist_genhlth_img, style={"width": "100%"}),
    html.H4("PhysHlth por Diabetes"),
    html.Img(src=hist_physhlth_img, style={"width": "100%"}),

    html.H3("Boxplots estratificados por classe (comparação robusta)"),
    html.H4("BMI por Diabetes"),
    html.Img(src=box_bmi_target_img, style={"width": "100%"}),
    html.H4("Age por Diabetes"),
    html.Img(src=box_age_target_img, style={"width": "100%"}),
    html.H4("GenHlth por Diabetes"),
    html.Img(src=box_genhlth_target_img, style={"width": "100%"}),
    html.H4("PhysHlth por Diabetes"),
    html.Img(src=box_physhlth_target_img, style={"width": "100%"}),
    html.H4("MentHlth por Diabetes"),
    html.Img(src=box_menthlth_target_img, style={"width": "100%"}),

    html.H2("5) Outliers (visão geral)"),
    html.Img(src=boxplot_img, style={"width": "100%"}),

    html.H2("6) Associação de variáveis binárias com diabetes (taxas)"),
    html.H4("HighBP → taxa média de diabetes"),
    html.Img(src=rate_highbp_img, style={"width": "80%"}),
    html.H4("HighChol → taxa média de diabetes"),
    html.Img(src=rate_highchol_img, style={"width": "80%"}),
    html.H4("PhysActivity → taxa média de diabetes"),
    html.Img(src=rate_physactivity_img, style={"width": "80%"}),

    html.H2("7) Correlações"),
    html.H3("Mapa de Correlação (Spearman)"),
    html.Img(src=heatmap_img, style={"width": "100%"}),

    html.H3(f"Correlação das variáveis com {target} (Spearman)"),
    html.Img(src=target_corr_img, style={"width": "100%"}),

])

if __name__ == '__main__':
    app.run(debug=True)
