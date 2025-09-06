import pandas as pd

# Carregar o arquivo (LEMBRE-SE DE USAR A OPÇÃO CORRETA PARA O CAMINHO!)
caminho = "C:/Users/julia/OneDrive/Área de Trabalho/IA Project/pre-processed.csv"
df = pd.read_csv(caminho)

# Explorar os dados
print("Primeiras 5 linhas:")
print(df.head())
print("\nEstrutura do DataFrame (colunas, tipos de dados):")
print(df.info())
print("\nContagem de labels (0=Fake, 1=True):")
print(df['label'].value_counts())
