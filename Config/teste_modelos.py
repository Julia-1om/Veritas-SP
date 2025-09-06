import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk
from nltk.corpus import stopwords

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

# 4. VETORIZAR OS TEXTOS - TESTANDO NOVOS PARÂMETROS
print("\nIniciando vetorização...")

# Teste diferentes valores para max_features e use ngram_range
vectorizer = TfidfVectorizer(
    stop_words=stop_words_portuguese,
    max_features=5000,  # Aumente o número de features (palavras consideradas)
    ngram_range=(1, 2)   # Considera palavras singles e pares (ex: "fake news")
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vetorização concluída!")

# 5. TREINAR O MODELO - TESTANDO SUAVIZAÇÃO
print("Treinando o modelo...")
model = MultinomialNB(alpha=0.1)  # Alfa menor = menos suavização
model.fit(X_train_vec, y_train)
print("Modelo treinado!")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Dicionário de modelos para testar
modelos = {
    'Naive Bayes': MultinomialNB(),
    'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, kernel='linear')
}

print("🤖 TESTANDO DIFERENTES MODELOS:")
print("=" * 50)

for nome, modelo in modelos.items():
    # Treinar e prever
    modelo.fit(X_train_vec, y_train)
    y_pred = modelo.predict(X_test_vec)
    acuracia = accuracy_score(y_test, y_pred)
    
    print(f"{nome:20} → Acurácia: {acuracia * 100:.2f}%")

    # Salvar o melhor modelo
    if acuracia > 0.90:  # Se acurácia for maior que 90%
        joblib.dump(modelo, f'modelo_{nome.replace(" ", "_")}.joblib')
        print(f"          💾 Modelo {nome} salvo!")