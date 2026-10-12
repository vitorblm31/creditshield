# CreditShield AI: Avaliador de Risco com Motor Neural

Um sistema *end-to-end* de Machine Learning desenvolvido para otimizar decisões de concessão de crédito, substituindo regras de corte engessadas por probabilidades dinâmicas e baseadas em dados.

## O Problema de Negócio
Instituições financeiras perdem capital de duas formas principais: aprovando crédito para clientes inadimplentes (*Falsos Positivos*) e negando empréstimos para bons pagadores devido a regras rígidas de pontuação (*Falsos Negativos*). O CreditShield AI foi desenhado para encontrar padrões não lineares no histórico financeiro, reduzindo o risco de *default* sem sacrificar a base de clientes saudáveis.

## A Solução Técnica
O motor preditivo foi construído com foco em interpretabilidade comercial, generalização e eficiência de processamento:
* **Dados:** Treinado sobre o *German Credit Data* (Kaggle), garantindo exposição a ruídos e variáveis categóricas do mundo real (20 *features* processadas).
* **Arquitetura (Deep Learning):** Multi-Layer Perceptron (MLP) construído do zero com **PyTorch**.
  * **Camadas Ocultas:** 32 e 16 neurônios com ativação não-linear ReLU.
  * **Regularização Avançada:** Aplicação de *Dropout* (20%) e *Weight Decay* (L2) no otimizador Adam para combater o *overfitting*, estabilizando a acurácia orgânica em dados invisíveis na marca de 77%.
  * **Camada de Saída:** Ativação Sigmoide para conversão direta do risco em probabilidade percentual (0 a 1).
* **Interface (MLOps):** *Dashboard* interativo em **Streamlit** (Smart Form). Permite que os operadores de crédito insiram apenas as 4 variáveis vitais da negociação na tela, enquanto o *backend* injeta automaticamente as 16 variáveis complementares com o perfil mediano do *dataset*, entregando o veredito instantaneamente sem atrito para o usuário.

## ⚙️ Como Executar Localmente

1. Clone o repositório e acesse a pasta do projeto:
```bash
git clone https://github.com/vitorblm31/creditshield.git
cd creditshield
