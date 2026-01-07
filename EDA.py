import pandas as pd
import io
import contextlib

import dash
from dash import html, dash_table

import matplotlib.pyplot as plt
import seaborn as sns
import base64


df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# transformando a variável diabetes em binaria
# db = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
# df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})
# target = 'Diabetes_binary'
# binary_col = [col for col in df.columns if df[col].nunique() == 2 and col != target]
# num_col = [col for col in df.columns.difference(binary_col) if col != target]

def get_df_info(df):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()

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

# transformando as variaveis em int
shape_text2 = f"Linhas: {db.shape[0]} | Colunas: {db.shape[1]}"
info_text2 = get_df_info(db)

nulls_df = db.isnull().sum().reset_index()
nulls_df.columns = ["Coluna", "Valores nulos"]

unique_values = {
    col: db[col].value_counts().shape[0]
    for col in db.columns
}
unique_df = pd.DataFrame.from_dict(
    unique_values, orient="index", columns=["Unique value count"]
).reset_index().rename(columns={"index": "Coluna"})

boxplot_img = generate_boxplot(db)


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={"padding": "20px"}, children=[

    html.H1("EDA – Diabetes Health Indicators Dataset"),

    html.H2("Link para o dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"),

    html.H2("Shape do Dataset"),
    html.P(shape_text),

    html.H2("Informações do Dataset"),
    html.Pre(info_text, style={"backgroundColor": "#f4f4f4", "padding": "10px"}),

    html.H2("Primeiras 5 linhas"),
    dash_table.DataTable(
        data=head_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in head_df.columns],
        page_size=5,
        style_table={"overflowX": "auto"}
    ),

    html.H2("Estatísticas Descritivas"),
    dash_table.DataTable(
        data=describe_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in describe_df.columns],
        style_table={"overflowX": "auto"}
    ),

    html.H2("Pre processamento - transformado as variaveis em int"),
    html.P(shape_text2),
    html.Pre(info_text2, style={"backgroundColor": "#f4f4f4", "padding": "10px"}),

    html.H2("Valores Nulos"),
    dash_table.DataTable(
        data=nulls_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in nulls_df.columns],
    ),

    html.H2("Valores Únicos por Atributo"),
    dash_table.DataTable(
        data=unique_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in unique_df.columns],
    ),

    html.H2("Duplicatas"),
    html.P(f"Antes: {duplicates_before} | Depois: {duplicates_after}"),

    html.H2("Análise de Outliers"),
    html.Img(src=boxplot_img, style={"width": "100%"})

])

if __name__ == '__main__':
    app.run(debug=True)