import os
import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="CreditShield AI", layout="centered")

class CreditShieldMLP(nn.Module):
    def __init__(self):
        super(CreditShieldMLP, self).__init__()
        self.camada_oculta1 = nn.Linear(20, 32)
        self.ativacao_relu1 = nn.ReLU()
        self.camada_oculta2 = nn.Linear(32, 16)
        self.ativacao_relu2 = nn.ReLU()
        self.camada_saida = nn.Linear(16, 1)
        self.ativacao_sigmoide = nn.Sigmoid()

    def forward(self, x):
        x = self.ativacao_relu1(self.camada_oculta1(x))
        x = self.ativacao_relu2(self.camada_oculta2(x))
        x = self.ativacao_sigmoide(self.camada_saida(x))
        return x

@st.cache_resource
def load_assets():
    caminho_scaler = os.path.join(BASE_DIR, 'credit_scaler.pkl')
    caminho_modelo = os.path.join(BASE_DIR, 'credit_model.pth')
    
    scaler = joblib.load(caminho_scaler)
    modelo = CreditShieldMLP()
    modelo.load_state_dict(torch.load(caminho_modelo))
    modelo.eval()
    return scaler, modelo

scaler, modelo = load_assets()

st.title("CreditShield AI - Avaliador de Risco")
st.markdown("### Avaliador de Risco com Motor Neural")
st.write("Insira os dados principais. O motor neural preencherá o perfil complementar automaticamente.")

with st.form("credit_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.number_input("Idade do Cliente (anos)", min_value=18, max_value=100, value=35)
        valor = st.number_input("Valor Solicitado (DM)", min_value=100, value=2500)
        
    with col2:
        duracao = st.number_input("Duração do Crédito (meses)", min_value=4, max_value=72, value=24)
        historico = st.selectbox("Histórico no Banco", [0, 1, 2, 3, 4], index=4, help="0: Crítico | 4: Excelente pagador")
        
    submit = st.form_submit_button("Avaliar Risco com IA", use_container_width=True)

if submit:
    entrada_bruta = np.array([[2, duracao, historico, 2, valor, 1, 3, 3, 3, 1, 3, 2, idade, 3, 2, 1, 3, 1, 1, 2]])
    entrada_escalada = scaler.transform(entrada_bruta)
    entrada_tensor = torch.FloatTensor(entrada_escalada)
    
    with torch.no_grad():
        probabilidade = modelo(entrada_tensor).item() * 100
        
    st.divider()
    if probabilidade > 50:
        st.error("⚠️ Risco Elevado de Incumprimento (Default)")
        st.write("Recomendação: **Negar Crédito** ou encaminhar para análise manual avançada.")
    else:
        st.success("✅ Risco Controlado (Cliente Saudável)")
        st.write("Recomendação: **Aprovar Crédito** conforme as políticas vigentes.")
        
    st.metric(label="Probabilidade Neural de Calote", value=f"{probabilidade:.1f}%")