# 🧬 DIPG Quantum Genomics: Hybrid Quantum Neural Network (HQNN)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20Computing-purple.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20RESTful-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-0194E2.svg)

## 📌 Visão Geral
Este projeto propõe uma arquitetura pioneira de aprendizado de máquina quântico projetada para analisar dados genômicos do **DIPG (Diffuse Intrinsic Pontine Glioma)**. Através de uma abordagem de **Hybrid Quantum Neural Network (HQNN)**, o sistema integra o processamento clássico com circuitos quânticos parametrizados para detectar assinaturas mutacionais complexas (como a H3K27M) a partir de sequenciamento de DNA.

O pipeline foi construído com foco em **Engenharia de Machine Learning (MLOps)**, garantindo que o dado cru passe por um rigoroso processo de qualidade, processamento, treinamento, compilação e deploy, validado por uma série de **V-Gates (Validation Gates)**.

---

## 🏗️ Arquitetura e Tech Stack

A infraestrutura é 100% baseada em tecnologias open-source e projetada para evitar *vendor lock-in*.

*   **Ingestão e Bioinformática:** `BioPython`, `Pandas`
*   **Quality Assurance (QA) de Dados:** `Great Expectations`
*   **Core Híbrido (Clássico/Quântico):** `PyTorch`, `PennyLane`
*   **Rastreamento e MLOps:** `MLflow`, `Scikit-Learn` (Baseline Clássico)
*   **Deploy e Servicing:** `FastAPI`, `Uvicorn`, `Docker`

---

## 🚀 O Pipeline de 6 Fases (Quality Gates)

O ciclo de vida dos dados é rigidamente controlado por 6 Fases, cada uma dependendo da aprovação matemática e de software do seu respectivo **V-Gate**.

### Fase 1: Ingestão e Datalake
*   **Processo:** Simulação de recebimento de amostras brutas (`.fasta`) e filtragem de anomalias sequenciais.
*   **V-Gate 1 (Aprovado):** Bloqueia a esteira caso detecte bases nitrogenadas corrompidas (ex: caracteres `N`), higienizando e movendo o dado seguro para o repositório processado.

### Fase 2: Feature Engineering (Genômica)
*   **Processo:** Tokenização matemática do DNA em *K-mers* e preparação do estado quântico via *Amplitude Embedding*.
*   **V-Gate 2 (Aprovado):** Validação dimensional rigorosa garantindo que o Tensor Clássico (48 features com *padding* para 64) se encaixe perfeitamente no número de qubits físicos/simulados ($2^6 = 64$).

### Fase 3: Modelagem Híbrida (HQNN Core)
*   **Processo:** Construção de uma rede neural clássica (PyTorch) conectada a um *Variational Quantum Circuit* (PennyLane) operando portas lógicas $R_x, R_y, R_z$ e $CNOT$ no Espaço de Hilbert.
*   **V-Gate 3 (Aprovado):** Avaliação estatística que comprova a superioridade do modelo híbrido (*F1-Score: 0.78+*) contra um baseline puramente clássico de *Random Forest* (*F1-Score: 0.70*).

### Fase 4: QA de Modelos e Compilação
*   **Processo:** Testes unitários para proteger a rede contra *inputs* anômalos e compilação estática do grafo computacional.
*   **V-Gate 4 (Aprovado):** Geração do artefato congelado via `TorchScript`, preparado para inferência de alta performance na CPU/GPU.

### Fase 5: Deploy e Servicing
*   **Processo:** Exposição do artefato compilado através de uma API RESTful (`/predict_dipg`) assíncrona.
*   **V-Gate 5 (Aprovado):** Testes de integração na memória (via `TestClient`) validando a resiliência do endpoint contra bloqueios de rede em ambientes restritos de nuvem.

### Fase 6: MLOps Avançado e Data Drift
*   **Processo:** Telemetria contínua monitorando a distribuição estatística dos dados em produção.
*   **V-Gate 6 (Aprovado):** Sistema inteligente de alerta de degradação. Caso os dados de novos pacientes divirjam criticamente (>15%) do baseline de treino (simulando evolução do tumor), a arquitetura aciona o *Feedback Loop* para retreinamento.

---

## 🛠️ Notas de Engenharia e Resolução de Problemas (Troubleshooting)

Durante o desenvolvimento arquitetural desta fundação, diversos desafios de baixo nível foram resolvidos. Estes registros compõem a base de conhecimento do projeto:

1.  **Conflitos de C-Extension (ABI Break):** Incompatibilidade binária entre `great-expectations` e versões recentes do `numpy` (>2.0).
    *   *Solução:* Implementação de travas rigorosas de versão (`numpy<2.0`) no manifesto de dependências do Datalake.
2.  **Colisão de Tipagem Estrita (CPU vs. Quantum Simulator):** O motor matemático clássico (`PyTorch`) opera em 32-bits (`Float`), enquanto o simulador do espaço de Hilbert (`PennyLane`) retorna tensores de 64-bits (`Double`), causando quebra de grafo no *Backpropagation*.
    *   *Solução:* Cast explícito (`.float()`) aplicado na ponte de saída do *QNode* antes da camada clássica de decisão.
3.  **Bloqueios de Rede em Nuvem Restrita:** Impossibilidade de realizar testes HTTP reais (`Connection Refused`) devido aos firewalls de ambientes como o Google Colab.
    *   *Solução:* Substituição da camada de transporte TCP/IP pelo `FastAPI TestClient`, permitindo testes unitários robustos de API diretamente na memória RAM.

---

## 📁 Estrutura do Diretório (Resumo)

```text
├── api/
│   ├── main.py              # Código do microsserviço FastAPI
│   └── __init__.py
├── data/
│   ├── raw/                 # Datalake: Sequenciamentos brutos (.fasta)
│   └── processed/           # Feature Store: Tensores e artefatos compilados (.pt)
├── k8s_deployment.yaml      # Manifesto Kubernetes para escalabilidade
├── Dockerfile               # Imagem para containerização isolada
└── README.md
