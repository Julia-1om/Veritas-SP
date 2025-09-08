import gradio as gr
import joblib
import pandas as pd
import re
import os

# Carregar modelo e vectorizer
model = joblib.load('modelo_svm.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Função de pré-processamento
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Função principal de classificação
def classificar_noticia(texto):
    if not texto.strip():
        return "Por favor, insira um texto para análise.", ""
    
    try:
        texto_limpo = preprocessar_texto(texto)
        texto_vetorizado = vectorizer.transform([texto_limpo])
        previsao = model.predict(texto_vetorizado)
        probabilidade = model.predict_proba(texto_vetorizado)
        
        if previsao[0] == 0:
            resultado = "🚫 POTENCIAL FAKE NEWS"
            confianca = f"{probabilidade[0][0] * 100:.2f}%"
        else:
            resultado = "✅ NOTÍCIA CONFIÁVEL"
            confianca = f"{probabilidade[0][1] * 100:.2f}%"
            
        return resultado, confianca
        
    except Exception as e:
        return f"Erro na análise: {str(e)}", ""

# Interface Gradio
interface = gr.Interface(
    fn=classificar_noticia,
    inputs=gr.Textbox(
        label="Cole o texto da notícia aqui:",
        placeholder="Ex: 'Nova descoberta revolucionária...'",
        lines=5
    ),
    outputs=[
        gr.Textbox(label="Resultado"),
        gr.Textbox(label="Confiança da Análise")
    ],
    title="🤖 Detector de Fake News em Português",
    description="Analise notícias com IA para identificar possíveis fake news. Desenvolvido com SVM (96.81% de acurácia).",
    examples=[
        ["Vacina contra COVID-19 contém chip para controlar a população"],
        ["Banco Central anuncia novas medidas econômicas para o próximo trimestre"]
    ]
)

# Iniciar a interface
if __name__ == "__main__":
    interface.launch()
