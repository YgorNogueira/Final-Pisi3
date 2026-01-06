import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('database.csv')

# transformando a variável diabetes em binaria
df = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})
target = 'Diabetes_binary'
binary_col = [col for col in df.columns if df[col].nunique() == 2 and col != target]
num_col = [col for col in df.columns.difference(binary_col) if col != target]

COLUMN_DESC = {
    "Diabetes_binary": "Variável alvo binária: 0 = não diabético, 1 = pré-diabetes ou diabetes (agrupado).",
    "HighBP": "Indica diagnóstico de pressão arterial elevada: 0 = sem pressão alta, 1 = com pressão alta.",
    "HighChol": "Indica diagnóstico de colesterol elevado: 0 = sem colesterol alto, 1 = com colesterol alto.",
    "CholCheck": "Indica se realizou exame de colesterol nos últimos 5 anos: 0 = não realizou, 1 = realizou.",
    "BMI": "Índice de Massa Corporal (IMC), valor numérico contínuo calculado por peso dividido pela altura ao quadrado.",
    "Smoker": "Indica se o indivíduo já fumou pelo menos 100 cigarros ao longo da vida: 0 = não, 1 = sim.",
    "Stroke": "Histórico de acidente vascular cerebral (derrame): 0 = nunca teve AVC, 1 = já teve AVC.",
    "HeartDiseaseorAttack": "Histórico de doença cardíaca ou ataque cardíaco: 0 = não, 1 = sim.",
    "PhysActivity": "Indica se praticou atividade física nos últimos 30 dias: 0 = não praticou, 1 = praticou.",
    "Fruits": "Indica consumo regular de frutas: 0 = não consome regularmente, 1 = consome regularmente.",
    "Veggies": "Indica consumo regular de vegetais: 0 = não consome regularmente, 1 = consome regularmente.",
    "HvyAlcoholConsump": "Indica consumo elevado de álcool (homens >14 doses/semana, mulheres >7 doses/semana): 0 = não, 1 = sim.",
    "AnyHealthcare": "Indica se possui algum tipo de acesso a serviços ou plano de saúde: 0 = não, 1 = sim.",
    "NoDocbcCost": "Indica se deixou de consultar um médico por motivos financeiros: 0 = não, 1 = sim.",
    "GenHlth": "Autoavaliação da saúde geral em escala ordinal de 1 a 5, onde valores maiores indicam pior percepção de saúde.",
    "MentHlth": "Número de dias nos últimos 30 em que a saúde mental não esteve boa, variando de 0 a 30.",
    "PhysHlth": "Número de dias nos últimos 30 em que a saúde física não esteve boa, variando de 0 a 30.",
    "DiffWalk": "Indica dificuldade para caminhar ou subir escadas: 0 = não possui dificuldade, 1 = possui dificuldade.",
    "Sex": "Sexo biológico do respondente: 0 = feminino, 1 = masculino.",
    "Age": "Faixa etária categorizada em valores de 1 a 13, representando intervalos crescentes de idade.",
    "Education": "Nível de escolaridade categorizado em escala de 1 a 6, onde valores maiores indicam maior escolaridade.",
    "Income": "Faixa de renda anual categorizada em escala de 1 a 8, onde valores maiores indicam maior renda."
}

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Análise Exploratória de Dados - Diabetes Health Indicators Dataset", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Seção 1: Visão Geral dos Dados
    html.Div([
        html.H2("1. Exploração dos Dados", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                html.H3("Estatísticas Descritivas"),
                html.P(f"Total de registros: {df.shape[0]:,}"),
                html.P(f"Total de variáveis: {df.shape[1]}"),
                html.P(f"Variáveis binárias: {len(binary_col)}"),
                html.P(f"Variáveis numéricas: {len(num_col)}"),
            ], className="six columns", style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 10}),
        ], className="row"),
        
        html.Div([
            html.H3("Amostra dos Dados"),
            html.Div(
                dash.dash_table.DataTable(
                    data=df.head(5).to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '5px',
                        'fontSize': '12px'
                    },
                    style_header={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'fontWeight': 'bold'
                    }
                )
            )
        ], style={'marginTop': 20})
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),

    html.Div([
        html.H2("📌 Visão por Variável", style={
            'color': '#3498db',
            'borderBottom': '2px solid #3498db',
            'paddingBottom': 10
        }),

        # Barra de seleção
        html.Div([
            html.Label("Selecione a variável:"),
            dcc.Dropdown(
                id="var-dropdown",
                options=[{"label": col, "value": col} for col in df.columns],
                value=df.columns[0],
                clearable=False
            )
        ], style={'marginBottom': 16}),

        # “Card” com título, descrição, histograma e dados
        html.Div([
            html.H3(id="var-title", style={'marginBottom': 6}),
            html.Div(id="var-desc", style={'color': '#7f8c8d', 'marginBottom': 14}),

            dcc.Graph(id="var-hist", config={"displayModeBar": False}),

            html.H4("Resultados (amostra)", style={'marginTop': 10}),
            dash_table.DataTable(
                id="var-table",
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '6px', 'fontSize': '12px'},
                style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'}
            ),

            html.Div(id="var-stats", style={'marginTop': 14})
        ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10})

    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),

], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'fontFamily': 'Arial, sans-serif'})
@app.callback(
    Output("var-title", "children"),
    Output("var-desc", "children"),
    Output("var-hist", "figure"),
    Output("var-table", "data"),
    Output("var-table", "columns"),
    Output("var-stats", "children"),
    Input("var-dropdown", "value"),
)
def update_variable_view(col):
    series = df[col].dropna()

    # Título e descrição
    title = f"# {col}"
    desc = COLUMN_DESC.get(col, "Sem descrição cadastrada para esta variável.")

    # Decide tipo de gráfico
    nunique = series.nunique()

    if nunique <= 10:
        # Discreto/binário -> contagem
        counts = series.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values))
        fig.update_layout(
            title=f"Distribuição de {col} (contagem)",
            xaxis_title=col,
            yaxis_title="Quantidade",
            template="plotly_white"
        )
    else:
        # Numérico contínuo -> histograma
        fig = px.histogram(series, nbins=min(30, nunique))
        fig.update_layout(
            title=f"Distribuição de {col} (histograma)",
            xaxis_title=col,
            yaxis_title="Frequência",
            template="plotly_white"
        )

    # Tabela de resultados (amostra)
    sample_df = df[[col]].head(10).copy()
    table_data = sample_df.to_dict("records")
    table_cols = [{"name": col, "id": col}]

    # Estatísticas rápidas
    stats_children = []

    stats_children.append(html.H4("Resumo rápido"))
    stats_children.append(html.P(f"Valores não nulos: {series.shape[0]:,}"))
    stats_children.append(html.P(f"Valores únicos: {nunique:,}"))

    if pd.api.types.is_numeric_dtype(series):
        stats_children.extend([
            html.P(f"Mínimo: {series.min():.3f}"),
            html.P(f"Mediana: {series.median():.3f}"),
            html.P(f"Média: {series.mean():.3f}"),
            html.P(f"Máximo: {series.max():.3f}")
        ])
    else:
        top = series.value_counts().head(5)
        stats_children.append(html.P("Top 5 valores:"))
        stats_children.append(html.Ul([html.Li(f"{idx}: {val}") for idx, val in top.items()]))

    return title, desc, fig, table_data, table_cols, stats_children

if __name__ == '__main__':
    app.run(debug=True)