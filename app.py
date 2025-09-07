# app.py
import streamlit as st
import pandas as pd
import re
import os

# Verificação de bibliotecas
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    import nltk
    from nltk.corpus import stopwords
except ImportError as e:
    st.error(f"❌ Erro ao importar bibliotecas: {e}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Detector de Fake News",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Detector de Fake News em Português")
st.write("Analisador de notícias com IA")

# Verificar se arquivos existem
if not os.path.exists('modelo_svm.joblib'):
    st.error("❌ Arquivo 'modelo_svm.joblib' não encontrado!")
    st.stop()

if not os.path.exists('vectorizer.joblib'):
    st.error("❌ Arquivo 'vectorizer.joblib' não encontrado!")
    st.stop()

# Função para pré-processar texto
def preprocessar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Carregar modelos com cache
@st.cache_resource
def carregar_modelos():
    try:
        model = joblib.load('modelo_svm.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelos: {e}")
        return None, None

# Carregar modelos
model, vectorizer = carregar_modelos()

if model is None or vectorizer is None:
    st.stop()

# Interface principal
texto_usuario = st.text_area(
    "Cole o texto da notícia aqui:",
    height=200,
    placeholder="Ex: 'Novo estudo revela descoberta revolucionária...'"
)

if st.button("Analisar Notícia 🔍", type="primary"):
    if not texto_usuario.strip():
        st.warning("⚠️ Por favor, insira um texto para análise.")
    else:
        with st.spinner("Analisando..."):
            try:
                texto_limpo = preprocessar_texto(texto_usuario)
                texto_vetorizado = vectorizer.transform([texto_limpo])
                previsao = model.predict(texto_vetorizado)
                probabilidade = model.predict_proba(texto_vetorizado)
                
                st.subheader("📊 Resultado:")
                if previsao[0] == 0:
                    st.error("🚫 **POTENCIAL FAKE NEWS**")
                    st.write(f"Confiança: {probabilidade[0][0]*100:.2f}%")
                else:
                    st.success("✅ **NOTÍCIA CONFIÁVEL**")
                    st.write(f"Confiança: {probabilidade[0][1]*100:.2f}%")
                    
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {e}")

# Rodapé
st.markdown("---")
st.caption("Projeto de IA - Detector de Fake News | Acuracia: 96.81%")
