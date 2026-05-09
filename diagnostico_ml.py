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
    modelo.fit(x_train, y_train)

    pred_train = modelo.predict(x_train)
    pred_test = modelo.predict(x_test)

    print(f"\n=== {nome} ===")
    print(f"Acurácia treino: {accuracy_score(y_train, pred_train):.3f}")
    print(f"Acurácia teste: {accuracy_score(y_test, pred_test):.3f}")
    print(f"Precisão teste: {precision_score(y_test, pred_test):.3f}")
    print(f"Recall teste: {recall_score(y_test, pred_test):.3f}")
    print(f"F1-score teste: {f1_score(y_test, pred_test):.3f}")
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, pred_test))

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(x_test[:5])
        print("Probabilidades das 5 primeiras amostras de teste:")
        print(probabilidades)


def main():
    dados = load_breast_cancer()
    x = dados.data
    y = dados.target

    print("Features:", dados.feature_names[:5], "...")
    print("Classes:", dados.target_names)
    print("Quantidade de exemplos:", x.shape[0])
    print("Quantidade de features:", x.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    regressao_logistica = LogisticRegression(max_iter=5000)
    arvore = DecisionTreeClassifier(random_state=42)

    avaliar_modelo(
        "Regressão logística",
        regressao_logistica,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    avaliar_modelo(
        "Árvore de decisão",
        arvore,
        x_train,
        x_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
