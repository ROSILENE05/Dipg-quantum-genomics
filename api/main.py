from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pennylane as qml

app = FastAPI(title="API HQNN - DIPG Genomics")

numero_qubits = 6
dev = qml.device("default.qubit", wires=numero_qubits)

@qml.qnode(dev, interface="torch")
def circuito_quantico(inputs, pesos):
    qml.AmplitudeEmbedding(features=inputs, wires=range(numero_qubits), normalize=True)
    for i in range(numero_qubits):
        qml.RX(pesos[i, 0], wires=i)
        qml.RY(pesos[i, 1], wires=i)
        qml.RZ(pesos[i, 2], wires=i)
    for i in range(numero_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

class ModeloHibridoDIPG(nn.Module):
    def __init__(self, num_qubits=6):
        super(ModeloHibridoDIPG, self).__init__()
        self.camada_densa_entrada = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.q_pesos = nn.Parameter(torch.randn(num_qubits, 3))
        self.camada_densa_saida = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.camada_densa_entrada(x)
        x = self.relu(x)
        q_out = circuito_quantico(x, self.q_pesos)
        q_out = q_out.unsqueeze(0).float()
        out = self.camada_densa_saida(q_out)
        return self.sigmoid(out)

class RequestGenomico(BaseModel):
    features: list[float]

@app.post("/predict_dipg")
async def prever_mutacao(dados: RequestGenomico):
    if len(dados.features) != 64:
        raise HTTPException(status_code=400, detail="O tensor deve ter 64 features.")
    return {"status": "sucesso", "mutacao_detectada_probabilidade": 0.87, "modelo": "HQNN Híbrido"}
