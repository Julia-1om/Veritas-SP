import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.svm import SVC  # Importamos apenas o SVM

# 1. CARREGAR OS DADOS
caminho = "C:/Users/julia/OneDrive/Área de Trabalho/IA Project/pre-processed.csv"
df = pd.read_csv(caminho)

# 2. PREPARAR OS DADOS
X = df['preprocessed_news']
y = df['label'].map({'fake': 0, 'true': 1})

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Treino: {len(X_train)} exemplos")
print(f"Teste:  {len(X_test)} exemplos")

# 3. BAIXAR STOPWORDS (fazer apenas uma vez)
nltk.download('stopwords')
stop_words_portuguese = stopwords.words('portuguese')

# 4. VETORIZAR OS TEXTOS
print("\nIniciando vetorização...")
vectorizer = TfidfVectorizer(
    stop_words=stop_words_portuguese,
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vetorização concluída!")

# 5. TREINAR E SALVAR O MELHOR MODELO (SVM) - REMOVEMOS O NAIVE BAYES
print("\nTreinando e salvando com (SVM)...")
melhor_modelo = SVC(random_state=42, kernel='linear', probability=True)
melhor_modelo.fit(X_train_vec, y_train)

# Avaliar final
y_pred = melhor_modelo.predict(X_test_vec)
acuracia_final = accuracy_score(y_test, y_pred)
print(f"Acurácia final do SVM: {acuracia_final * 100:.2f}%")

# Relatório de classificação detalhado
print("\nRELATÓRIO DE CLASSIFICAÇÃO (SVM):")
print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))

# 6. SALVAR O MODELO E O VECTORIZER PARA O STREAMLIT
joblib.dump(melhor_modelo, 'modelo_svm.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print("Modelo SVM e vectorizer salvos!")

# 7. ANÁLISE DOS ERROS (OPCIONAL, MAS MUITO INTERESSANTE PARA A APRESENTAÇÃO)
df_test = X_test.copy()
df_test = pd.DataFrame(df_test)
df_test['label_real'] = y_test.values
df_test['label_previsto'] = y_pred
df_test['texto'] = X_test.values

erros = df_test[df_test['label_real'] != df_test['label_previsto']]
print(f"\nNúmero de erros: {len(erros)}")

if not erros.empty:
    print("\nExemplo de um erro:")
    print("Texto:", erros.iloc[0]['texto'])
    print(f"Real: {'Fake' if erros.iloc[0]['label_real'] == 0 else 'True'}, Previsto: {'Fake' if erros.iloc[0]['label_previsto'] == 0 else 'True'}")