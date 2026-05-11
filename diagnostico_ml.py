"""Diagnóstico de retomada em Aprendizado de Máquina.

Complete os TODOs, rode o arquivo e copie os resultados principais para o
README.md.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def avaliar_modelo(nome, modelo, x_train, x_test, y_train, y_test):
    """Treina, prediz e imprime métricas simples."""
    # TODO 5: treine o modelo com os dados de treino.

    # TODO 6: gere as previsões para treino e teste.

    # TODO 7: imprima acurácia de treino, acurácia de teste, precisão, recall,
    # F1-score e matriz de confusão usando as funções importadas acima.

    # TODO 8: se o modelo tiver predict_proba, mostre as probabilidades das
    # cinco primeiras amostras de teste.

    raise NotImplementedError("Complete os TODOs da função avaliar_modelo.")


def main():
    # TODO 1: carregue o dataset load_breast_cancer().

    # TODO 2: separe os dados de entrada em X e o alvo em y.

    # TODO 3: imprima informações básicas: nomes das primeiras features,
    # classes, quantidade de exemplos e quantidade de features.

    # TODO 4: divida X e y em treino e teste com train_test_split.
    # Use test_size=0.25, random_state=42 e stratify=y.

    # TODO 9: crie uma regressão logística e uma árvore de decisão.

    # TODO 10: chame avaliar_modelo para os dois modelos.

    raise NotImplementedError("Complete os TODOs da função main.")


if __name__ == "__main__":
    main()
