import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

caminho_csv = os.path.join(BASE_DIR, 'german_credit.csv')
df = pd.read_csv(caminho_csv)

df['calote'] = 1 - df['kredit']
df = df.drop('kredit', axis=1)

X = df.drop('calote', axis=1).values
y = df['calote'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

class CreditShieldMLP(nn.Module):
    def __init__(self):
        super(CreditShieldMLP, self).__init__()
        self.camada_oculta1 = nn.Linear(20, 32)
        self.ativacao_relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.camada_oculta2 = nn.Linear(32, 16)
        self.ativacao_relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.camada_saida = nn.Linear(16, 1)
        self.ativacao_sigmoide = nn.Sigmoid()

    def forward(self, x):
        x = self.ativacao_relu1(self.camada_oculta1(x))
        x = self.dropout1(x)
        
        x = self.ativacao_relu2(self.camada_oculta2(x))
        x = self.dropout2(x)
        
        x = self.ativacao_sigmoide(self.camada_saida(x))
        return x

modelo = CreditShieldMLP()

criterio = nn.BCELoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)

epocas = 400 

print("Treinando o CreditShield com Otimizações...\n")

modelo.train()
for epoca in range(epocas):
    previsoes = modelo(X_train_tensor)
    erro = criterio(previsoes, y_train_tensor)
    
    otimizador.zero_grad()
    erro.backward()
    otimizador.step()

modelo.eval()
with torch.no_grad():
    previsoes_teste = modelo(X_test_tensor)
    previsoes_finais = (previsoes_teste >= 0.5).float()
    acertos = (previsoes_finais == y_test_tensor).sum()
    acuracia = acertos / y_test_tensor.shape[0]
    print(f'Acurácia: {acuracia * 100:.2f}%')

caminho_scaler = os.path.join(BASE_DIR, 'credit_scaler.pkl')
caminho_modelo = os.path.join(BASE_DIR, 'credit_model.pth')

joblib.dump(scaler, caminho_scaler)
torch.save(modelo.state_dict(), caminho_modelo)
print("\nRecursos otimizados guardados com sucesso!")