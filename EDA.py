import pandas as pd
import io
import contextlib
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import dash
from dash import html, dash_table
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Configurações visuais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.2f}'.format)


def get_df_info(df):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()

def fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_target_distribution(df, target="Diabetes_binary"):
    """Distribuição do target com gráfico de barras e pizza."""
    counts = df[target].value_counts().sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de barras
    sns.countplot(data=df, x=target, ax=axes[0])
    axes[0].set_title('Contagem por Classe')
    axes[0].set_xlabel('Diabetes (0=Não, 1=Sim)')
    axes[0].set_ylabel('Contagem')
    
    # Adicionar valores nas barras
    for i, count in enumerate(counts.values):
        axes[0].text(i, count + 0.02*max(counts.values), 
                    f'{count:,}\n({count/len(df)*100:.1f}%)', 
                    ha='center', va='bottom')
    
    # Gráfico de pizza
    axes[1].pie(counts.values, labels=['Sem Diabetes', 'Com Diabetes'], 
                autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    axes[1].set_title('Proporção por Classe')
    
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
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x=target, y=col)
    plt.title(f"{col} por {target}")
    plt.xlabel(target)
    plt.ylabel(col)
    return fig_to_base64()

def generate_binary_rate_plot(df, col, target="Diabetes_binary"):
    rate = df.groupby(col)[target].mean().reset_index()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(rate[col].astype(str), rate[target]*100, color=['lightblue', 'salmon'])
    plt.title(f"Taxa média de {target} por {col}")
    plt.xlabel(col)
    plt.ylabel(f"Taxa de {target} (%)")
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    return fig_to_base64()

def generate_outliers_analysis(df, numeric_vars=['BMI', 'MentHlth', 'PhysHlth', 'Age']):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    outliers_info = {}
    
    for i, var in enumerate(numeric_vars):
        sns.boxplot(data=df, y=var, ax=axes[i], color='lightblue')
        axes[i].set_title(f'Boxplot de {var}')
        axes[i].set_ylabel(var)
        
        # Calcular estatísticas de outliers
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
        outliers_info[var] = {
            'count': len(outliers),
            'percent': len(outliers)/len(df)*100,
            'bounds': [lower_bound, upper_bound]
        }
    
    plt.tight_layout()
    return fig_to_base64(), outliers_info

def generate_histograms_distribution(df, numeric_vars=['BMI', 'MentHlth', 'PhysHlth', 'Age']):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(numeric_vars):
        sns.histplot(data=df, x=var, ax=axes[i], kde=True, bins=30)
        axes[i].set_title(f'Distribuição de {var}')
        axes[i].set_xlabel(var)
        
        # Adicionar linhas de média e mediana
        mean_val = df[var].mean()
        median_val = df[var].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
        axes[i].axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    return fig_to_base64()

def generate_group_comparison_stats(df, target="Diabetes_binary", comparison_vars=['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth']):
    df_diabetes = df[df[target] == 1]
    df_no_diabetes = df[df[target] == 0]
    
    comparison_df = pd.DataFrame()
    
    for var in comparison_vars:
        mean_no = df_no_diabetes[var].mean()
        mean_yes = df_diabetes[var].mean()
        diff = mean_yes - mean_no
        diff_perc = (diff / mean_no) * 100 if mean_no != 0 else 0
        
        # Teste t para verificar significância
        t_stat, p_value = stats.ttest_ind(df_no_diabetes[var].dropna(), 
                                         df_diabetes[var].dropna())
        
        comparison_df[var] = {
            'Sem Diabetes': f"{mean_no:.2f}",
            'Com Diabetes': f"{mean_yes:.2f}",
            'Diferença': f"{diff:.2f}",
            '% Diferença': f"{diff_perc:.1f}%",
            'p-value': f"{p_value:.4f}",
            'Significativo': 'SIM' if p_value < 0.05 else 'NÃO'
        }
    
    return comparison_df.T

def generate_group_comparison_plots(df, target="Diabetes_binary", comparison_vars=['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth']):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(comparison_vars[:6]):
        sns.boxplot(data=df, x=target, y=var, ax=axes[i])
        axes[i].set_title(f'Distribuição de {var} por Grupo')
        axes[i].set_xlabel('Diabetes (0=Não, 1=Sim)')
        axes[i].set_ylabel(var)
    
    # Remover eixos extras
    for i in range(len(comparison_vars[:6]), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig_to_base64()

def generate_density_plots(df, target="Diabetes_binary", main_vars=['BMI', 'Age', 'GenHlth', 'PhysHlth']):
    df_diabetes = df[df[target] == 1]
    df_no_diabetes = df[df[target] == 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(main_vars):
        # Histograma sobreposto
        sns.histplot(data=df_no_diabetes, x=var, label='Sem Diabetes', 
                     kde=True, alpha=0.5, ax=axes[i], color='blue')
        sns.histplot(data=df_diabetes, x=var, label='Com Diabetes', 
                     kde=True, alpha=0.5, ax=axes[i], color='red')
        axes[i].set_title(f'Distribuição de {var} por Grupo')
        axes[i].set_xlabel(var)
        axes[i].legend()
    
    plt.tight_layout()
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
    bars = plt.barh(range(len(correlationSeries)), correlationSeries.values,
                   color=['red' if x > 0 else 'blue' for x in correlationSeries.values])
    plt.yticks(range(len(correlationSeries)), correlationSeries.index)
    plt.xlabel('Coeficiente de Correlação de Spearman')
    plt.title(f'Correlação com {target}')
    plt.grid(axis='x', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center', ha='left' if width >= 0 else 'right')
    
    return fig_to_base64()

def generate_binary_vars_analysis(df, target="Diabetes_binary", binary_vars=['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 
                                                                            'HvyAlcoholConsump', 'DiffWalk', 'HeartDiseaseorAttack']):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(binary_vars):
        if i < len(axes):
            # Calcular taxas
            rates = df.groupby(var)[target].mean() * 100
            categories = ['Não', 'Sim'] if len(rates) == 2 else rates.index
            
            # Gráfico de barras
            bars = axes[i].bar(categories, rates.values, 
                              color=['lightblue', 'salmon'][:len(rates)])
            axes[i].set_title(f'Taxa de Diabetes por {var}')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Taxa de Diabetes (%)')
            axes[i].grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')
    
    # Remover eixos extras
    for i in range(len(binary_vars), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig_to_base64()

def generate_pairplot_sample(df, target="Diabetes_binary", sample_size=5000):
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    pairplot = sns.pairplot(data=sample_df, vars=['BMI', 'Age', 'GenHlth'], 
                            hue=target, palette={0: 'blue', 1: 'red'},
                            plot_kws={'alpha': 0.6}, diag_kind='kde')
    pairplot.fig.suptitle('Relações entre Variáveis Principais', y=1.02)
    
    buf = io.BytesIO()
    pairplot.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_scatter_plot(df, target="Diabetes_binary", sample_size=5000):
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(sample_df['BMI'], sample_df['Age'], 
                          c=sample_df[target], cmap='coolwarm', 
                          alpha=0.6, s=20)
    plt.xlabel('BMI')
    plt.ylabel('Idade')
    plt.title('Relação entre BMI e Idade (cor = diabetes)')
    plt.colorbar(scatter, label='Diabetes (0=Não, 1=Sim)')
    plt.grid(alpha=0.3)
    
    return fig_to_base64()

def build_group_stats_table(df, target="Diabetes_binary", cols=None):
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

def generate_boxplot(df):
    plt.figure(figsize=(15, 15))
    for i, col in enumerate(['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age','Education', 'Income']):
        plt.subplot(4, 2, i + 1)
        sns.boxplot(x=col, data=df)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# Carregar dataset
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

shape_original = f"Dataset original: {df.shape[0]} linhas × {df.shape[1]} colunas"

info_text = get_df_info(df)

head_df = df.head(5)

describe_df = df.describe().reset_index()
desc_stats = df.describe().T
desc_stats['cv'] = desc_stats['std'] / desc_stats['mean']

nulls_df = df.isnull().sum().reset_index()
nulls_df.columns = ["Coluna", "Valores nulos"]

duplicates_before = df.duplicated().sum()

# Pré-processamento
db = df.copy()
for col in db.columns:
    db[col] = db[col].astype(int)

db.drop_duplicates(inplace=True)
duplicates_after = db.duplicated().sum()

shape_processado = f"Dataset após pré-processamento: {db.shape[0]} linhas × {db.shape[1]} colunas"
info_text2 = get_df_info(db)

unique_values = {col: db[col].nunique() for col in db.columns}
unique_df = pd.DataFrame.from_dict(unique_values, orient="index", columns=["Unique value count"]) \
    .reset_index().rename(columns={"index": "Coluna"})

target = "Diabetes_binary"

# Análise do target
target_dist_img = generate_target_distribution(db, target=target)
target_dist_counts = db[target].value_counts()
target_dist_perc = db[target].value_counts(normalize=True) * 100

# Análise de outliers
outliers_img, outliers_info = generate_outliers_analysis(db)
boxplot_img_outliers = generate_boxplot(db)

# Histogramas de distribuição
hist_dist_img = generate_histograms_distribution(db)

# Comparação entre grupos
comparison_stats_df = generate_group_comparison_stats(db, target=target)
group_comparison_img = generate_group_comparison_plots(db, target=target)
density_plots_img = generate_density_plots(db, target=target)

# Tabela de médias/medianas
group_stats_df = build_group_stats_table(db, target=target)

# Histogramas por classe
hist_bmi_img = generate_numeric_hist_by_target(db, "BMI", target=target, bins=35)
hist_age_img = generate_numeric_hist_by_target(db, "Age", target=target, bins=13)
hist_genhlth_img = generate_numeric_hist_by_target(db, "GenHlth", target=target, bins=5)
hist_physhlth_img = generate_numeric_hist_by_target(db, "PhysHlth", target=target, bins=31)

# Boxplots por classe
box_bmi_target_img = generate_numeric_box_by_target(db, "BMI", target=target)
box_age_target_img = generate_numeric_box_by_target(db, "Age", target=target)
box_genhlth_target_img = generate_numeric_box_by_target(db, "GenHlth", target=target)
box_physhlth_target_img = generate_numeric_box_by_target(db, "PhysHlth", target=target)
box_menthlth_target_img = generate_numeric_box_by_target(db, "MentHlth", target=target)

# Taxas de variáveis binárias
rate_highbp_img = generate_binary_rate_plot(db, "HighBP", target=target)
rate_highchol_img = generate_binary_rate_plot(db, "HighChol", target=target)
rate_physactivity_img = generate_binary_rate_plot(db, "PhysActivity", target=target)
binary_vars_img = generate_binary_vars_analysis(db, target=target)

# Análise de correlação
heatmap_img = generateHeatMap(db, method="spearman", show_annot=False)
target_corr_img = generateTargetCorrelationPlot(db, target=target, method="spearman")

# Análise multivariada
pairplot_img = generate_pairplot_sample(db, target=target)
scatter_img = generate_scatter_plot(db, target=target)

# Coeficiente de variação
cv_df = desc_stats[['cv']].reset_index()
cv_df.columns = ['Variável', 'Coeficiente de Variação']
cv_df = cv_df.sort_values('Coeficiente de Variação', ascending=False)

# Salvar dataset processado
db.to_csv('diabetes_processed.csv', index=False)
print(f"Dataset processado salvo como 'diabetes_processed.csv'")


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={"padding": "20px", "fontFamily": "Arial, sans-serif"}, children=[

    html.H1("EDA Completo – Diabetes Health Indicators Dataset", style={'color': '#2c3e50'}),
    html.H2("Link para o dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset", 
            style={'color': '#3498db', 'fontSize': '16px'}),

    html.H2("1) Visão geral do Dataset", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.H3("Shape do Dataset"),
    html.P(shape_original, style={'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px'}),
    
    html.H3("Informações do Dataset"),
    html.Pre(info_text, style={"backgroundColor": "#f8f9fa", "padding": "15px", "border": "1px solid #dee2e6",
                               "borderRadius": "5px", "overflowX": "auto"}),

    html.H3("Primeiras 5 linhas"),
    dash_table.DataTable(
        data=head_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in head_df.columns],
        page_size=5,
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
        ]
    ),

    html.H3("Estatísticas Descritivas"),
    dash_table.DataTable(
        data=describe_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in describe_df.columns],
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        page_size=10
    ),

    html.H2("2) Pré-processamento básico", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.H3("Conversão para int + remoção de duplicatas"),
    html.P(shape_processado, style={'backgroundColor': '#d5f4e6', 'padding': '10px', 'borderRadius': '5px'}),
    html.Pre(info_text2, style={"backgroundColor": "#f8f9fa", "padding": "15px", "border": "1px solid #dee2e6",
                               "borderRadius": "5px", "overflowX": "auto"}),

    html.H3("Valores Nulos"),
    dash_table.DataTable(
        data=nulls_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in nulls_df.columns],
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'}
    ),

    html.H3("Valores Únicos por Atributo"),
    dash_table.DataTable(
        data=unique_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in unique_df.columns],
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        page_size=15
    ),

    html.H3("Duplicatas"),
    html.P(f"Antes: {duplicates_before} | Depois: {duplicates_after}", 
           style={'backgroundColor': '#ffeaa7', 'padding': '10px', 'borderRadius': '5px'}),

    html.H2("3) Coeficiente de Variação", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    html.P("Medida de dispersão relativa (CV = desvio padrão / média)", style={'fontStyle': 'italic'}),
    
    dash_table.DataTable(
        data=cv_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in cv_df.columns],
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        page_size=10
    ),

    html.H2("4) Análise de Outliers", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.H3("Boxplots para detecção de outliers"),
    html.Img(src=boxplot_img_outliers, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),
    
    html.H3("Análise detalhada de outliers (BMI, MentHlth, PhysHlth, Age)"),
    html.Img(src=outliers_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),
    
    html.H4("Resumo dos Outliers por Variável:"),
    html.Ul([
        html.Li(f"{var}: {info['count']} outliers ({info['percent']:.2f}%) - Limites: [{info['bounds'][0]:.2f}, {info['bounds'][1]:.2f}]")
        for var, info in outliers_info.items()
    ]),

    html.H2("5) Distribuições das Variáveis Principais", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    html.Img(src=hist_dist_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H2("6) Distribuição do alvo (balanceamento)", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    html.P([
        f"Sem diabetes: {target_dist_counts[0]:,} ({target_dist_perc[0]:.1f}%) | ",
        f"Com diabetes: {target_dist_counts[1]:,} ({target_dist_perc[1]:.1f}%)"
    ], style={'backgroundColor': '#dfe6e9', 'padding': '10px', 'borderRadius': '5px'}),
    html.Img(src=target_dist_img, style={"width": "80%", "margin": "0 auto", "display": "block", 
                                         "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H2("7) Comparação por classe (Sem diabetes vs Com diabetes)", 
            style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.H3("Teste estatístico de diferenças entre grupos"),
    html.P("Teste t de Student para comparar médias (p-value < 0.05 indica diferença significativa)", 
           style={'fontStyle': 'italic'}),
    
    dash_table.DataTable(
        data=comparison_stats_df.reset_index().rename(columns={'index': 'Variável'}).to_dict("records"),
        columns=[{"name": i, "id": i} for i in comparison_stats_df.reset_index().rename(columns={'index': 'Variável'}).columns],
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_data_conditional=[
            {'if': {'column_id': 'Significativo', 'filter_query': '{Significativo} = "SIM"'},
             'backgroundColor': '#d4edda', 'color': '#155724'},
            {'if': {'column_id': 'Significativo', 'filter_query': '{Significativo} = "NÃO"'},
             'backgroundColor': '#f8d7da', 'color': '#721c24'}
        ]
    ),

    html.H3("Boxplots de comparação entre grupos"),
    html.Img(src=group_comparison_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H3("Densidade por grupo"),
    html.Img(src=density_plots_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H3("Tabela de médias e medianas por classe"),
    dash_table.DataTable(
        data=group_stats_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in group_stats_df.columns],
        style_table={"overflowX": "auto", "border": "1px solid #dee2e6"},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        page_size=10
    ),

    html.H3("Histogramas por classe (principais)"),
    
    html.Div([
        html.Div([
            html.H4("BMI por Diabetes"),
            html.Img(src=hist_bmi_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("Age por Diabetes"),
            html.Img(src=hist_age_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"})
    ]),
    
    html.Div([
        html.Div([
            html.H4("GenHlth por Diabetes"),
            html.Img(src=hist_genhlth_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("PhysHlth por Diabetes"),
            html.Img(src=hist_physhlth_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"})
    ]),

    html.H3("Boxplots estratificados por classe (comparação robusta)"),
    
    html.Div([
        html.Div([
            html.H4("BMI por Diabetes"),
            html.Img(src=box_bmi_target_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("Age por Diabetes"),
            html.Img(src=box_age_target_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"})
    ]),
    
    html.Div([
        html.Div([
            html.H4("GenHlth por Diabetes"),
            html.Img(src=box_genhlth_target_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "32%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("PhysHlth por Diabetes"),
            html.Img(src=box_physhlth_target_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "32%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("MentHlth por Diabetes"),
            html.Img(src=box_menthlth_target_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "32%", "display": "inline-block", "padding": "10px"})
    ]),

    html.H2("8) Associação de variáveis binárias com diabetes (taxas)", 
            style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.Div([
        html.Div([
            html.H4("HighBP → taxa média de diabetes"),
            html.Img(src=rate_highbp_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "32%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("HighChol → taxa média de diabetes"),
            html.Img(src=rate_highchol_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "32%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.H4("PhysActivity → taxa média de diabetes"),
            html.Img(src=rate_physactivity_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"})
        ], style={"width": "32%", "display": "inline-block", "padding": "10px"})
    ]),
    
    html.H4("Análise completa de variáveis binárias"),
    html.Img(src=binary_vars_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H2("9) Correlações", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.H3("Mapa de Correlação (Spearman)"),
    html.Img(src=heatmap_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H3(f"Correlação das variáveis com {target} (Spearman)"),
    html.Img(src=target_corr_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),

    html.H2("10) Análise Multivariada", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
    
    html.H3("Pairplot - Relações entre variáveis principais (amostra de 5,000 observações)"),
    html.P("Azul = Sem diabetes, Vermelho = Com diabetes", style={'fontStyle': 'italic'}),
    html.Img(src=pairplot_img, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "5px"}),
    
    html.H3("Scatter Plot - BMI vs Idade (cor = diabetes)"),
    html.Img(src=scatter_img, style={"width": "80%", "margin": "0 auto", "display": "block", 
                                     "border": "1px solid #ddd", "borderRadius": "5px"}),


])

if __name__ == '__main__':
    app.run(debug=True)