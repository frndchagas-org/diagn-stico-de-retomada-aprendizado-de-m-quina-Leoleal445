[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ARkoM8Jo)
# Diagnóstico de retomada - Aprendizado de Máquina

Esta atividade serve para mapear o que você já domina em Aprendizado de Máquina depois das atividades anteriores da disciplina.

Responda individualmente. Use suas palavras. Rode o código quando possível. Se usar IA depois da primeira tentativa, registre o uso na seção 8.

Prazo: 11/05/2026 às 23:59, horário de Fortaleza.

## 1. Mapa do que eu lembro

Marque cada tópico como: lembro bem, lembro parcialmente, não lembro, nunca vi ou não tenho certeza.

- vetores, matrizes e produto escalar:lembro bem
- média, desvio padrão e correlação:lembro bem
- probabilidade condicional e Teorema de Bayes:lembro parcialmente
- regressão linear:lembro bem
- classificação supervisionada:lembro parcialmente
- treino, teste e validação:lembro bem
- normalização ou padronização de dados:pouco ou quase nada
- KNN:não lembro
- árvore de decisão:lembro parcialmente
- matriz de confusão:lembro parcialmente
- acurácia, precisão, recall e F1-score:lembro bem
- overfitting e underfitting:lembro parcialmente
- validação cruzada:lembro parcialmente
- Random Forest:lembro bem
- XGBoost ou boosting:lembro parcialmente
- `predict_proba()`:lembro parcialmente
- SQL/ETL aplicado a dados:pouco
- simulação de Monte Carlo:lembro bem

## 2. O que foi trabalhado antes

Explique, em 8 a 12 linhas:

1. quais desses tópicos você lembra de ter trabalhado na disciplina;
2. quais atividades ou exemplos você lembra;
3. o que você conseguiu fazer com autonomia;
4. o que você só conseguiu fazer seguindo roteiro;
5. qual assunto precisa ser retomado com mais urgência.
O professor sempre buscou mostrar como a área pode transformar dados em informações, com base nisso ele nos desafiou importar datasets e organizar a exploração e o alcance das informações. Um desses desafios foi no trabalho sobre a previsão do preço de aluguel de imóveis com base em aspectos como localização e outros fatores. Além disso, também foi apresentado outro desafio para criar um modelo de predição de resultados de partidas de futebol com base em Random Forest ou XGBoost por meio de um banco de dados SQLite, e utilizá-lo para prever o vencedor de um campeonato com base na simulação de Monte Carlo. Com autonomia foi possível construir a base do treinamento dos modelos (separar os dados em teste e treino, obter estatísticas de desempenho, etc). Essa etapa mostrou que o desempenho de uma variável analisada é diretamente propocional a a sua magnitude. Um ponto difícil na hora de executar as tarefas foi trabalhar com os bancos de dados relacionais e tarefas mais estatísticas (como a realização de simulação de Monte Carlo), apesar do trabalho ter sido em grupo às vezes funções x e y eram direcionadas especificamente a cada um.
## 3. Conceitos essenciais

Responda com suas palavras e dê um exemplo simples.

1. O que é aprendizado supervisionado?
Um exemplo didático que o professor sempre oferecia para explicar esse tipo de aprendizado era o comportamento e treinamento dado a um cão farejador. O cão farejador é treinado por um humano que apresenta o estímulo (o cheiro) e confirma ou corrige a resposta do animal com reforço positivo ou negativo. Ou seja, o "rotulador" é o treinador, ele fornece o gabarito.
2. O que é uma tarefa de classificação?
É a capacidade do sistema gerar uma classificação com base nos atributos, ou seja, se uma laranja é de fato uma laranja, ou uma maçã é uma maçã. Cor, rugosidade e outros fatores são indicadores. 
3. O que são features e target?
Features são as características das classes a serem definidas (Ex: peso, cor, formato) e o Target é o rótulo que será dado pelo modelo.
4. Para que serve separar treino e teste?
Evitar o overfitting, que pode causar uma "viés" no modelo, prejudicando sua capacidade de previsão.
5. O que é overfitting?
Esse fenomêno é correspondente a um contexto em que o modelo "decora" invés de aprender, o professor forneceu exemplos de quando um aluno decora questões de uma prova com base em provas antigas e não se prepara para possíveis variações. 
6. Por que acurácia pode ser uma métrica enganosa?
A acurácia não é coerente quanto a acertos e erros. O professor apresentava exemplos em cenários críticos, como na saúde, em que um falso positivo é menos grave do que ignorar um diagnóstico. Em um cenário de que o modelo esteja prevendo diabetes em um conjunto de pacientes de uma Unidade de Pronto Atendimento Imagine, onde 90 pacientes são saudáveis e 10 têm diabetes, um modelo que responde "sem diabetes" para todo mundo acerta 90% e parece ótimo, mas errou todos os 10 diabéticos, que vão embora da UPA sem diagnóstico e sem tratamento. 
## 4. Diagnóstico prático com Scikit-Learn

No arquivo `diagnostico_ml.py`, use o dataset `load_breast_cancer` do Scikit-Learn e faça:

1. carregue os dados;
2. separe `X` e `y`;
3. divida em treino e teste;
4. treine uma regressão logística;
5. treine uma árvore de decisão;
6. mostre matriz de confusão, acurácia, precisão, recall e F1-score para cada modelo;
7. compare o desempenho em treino e teste;
8. escreva aqui qual modelo generalizou melhor e por quê.

Se não conseguir terminar tudo, registre até onde chegou e qual erro apareceu.

### Resultados

Cole aqui os principais resultados do seu código.
=== Regressão logística ===
Acurácia treino: 0.958
Acurácia teste: 0.958
Precisão teste: 0.947
Recall teste: 0.989
F1-score teste: 0.967
Matriz de confusão:
[[48  5]
 [ 1 89]]
Probabilidades das 5 primeiras amostras de teste:
[[0.01854698 0.98145302]
 [0.99816861 0.00183139]
 [0.17716357 0.82283643]
 [0.2344741  0.7655259 ]
 [0.19796601 0.80203399]]

=== Árvore de decisão ===
Acurácia treino: 1.000
Acurácia teste: 0.923
Precisão teste: 0.954
Recall teste: 0.922
F1-score teste: 0.938
Matriz de confusão:
[[49  4]
 [ 7 83]]
Probabilidades das 5 primeiras amostras de teste:
[[0. 1.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [0. 1.]]  

### Interpretação

Qual modelo generalizou melhor? Explique usando as métricas e a comparação entre treino e teste.
O modelo de Regressão Logística generalizou melhor. Ficou claro que o modelo de Árvore de Decisão sofreu overfitting, apresentando uma acurácia de 100%, mas geralmente ajuda pior nos outros testes em comparação com a Regressão Logística (principalmente em F1).

## 5. Probabilidade e interpretação

Escolha um dos modelos treinados e responda:

1. O modelo produz probabilidade com `predict_proba()`?
2. O que significa uma probabilidade alta para uma classe?
3. Probabilidade alta garante que a previsão está correta? Explique.
4. Em um problema real, qual seria o risco de confiar cegamente nessa previsão?

Resposta:
Tomando a Regressão Logística como exemplo pode ser conluído que o modelo disponibiliza probabilidades através do predict_proba(). Uma probabilidade alta para uma determinada classe significa que o modelo avaliou aquela amostra como tendo grande chance de pertencer àquele grupo. Porém, isso não é uma garantia de acerto, é apenas o grau de confiança interna do modelo, que pode estar calibrado de forma errada ou ter sido treinado com dados que não representam bem a realidade. Em um problema real, como triagem de pacientes ou concessão de crédito, depositar confiança cega nesse número pode gerar decisões equivocadas com consequências sérias, tendo em mente que um modelo pode estar 95% "confiante" e ainda assim errar, especialmente em casos fora do padrão que ele aprendeu.
## 6. Generalização

Compare treino e teste:

1. Há sinal de overfitting?
2. Há sinal de underfitting?
3. O que você tentaria mudar para melhorar o resultado?
4. O que você precisaria estudar melhor para responder com mais segurança?

Resposta:
Se tratando da Regressão Logística, não parece haver overfitting e nem underfitting. Para melhorar o resultado, talvez adicionar mais recursos e manipular hiperparâmetros. Para responder com mais segurança, você precisa entender melhor como funcionam os hiperparâmetros de cada modelo.
## 7. Ponto de dificuldade

Escolha um tópico da lista inicial e escreva:

1. o que você entende dele;
2. onde você se confunde;
3. que tipo de explicação ajudaria: exemplo no quadro, notebook guiado, exercício curto, revisão matemática, visualização ou projeto pequeno.

Resposta:
Entendo que o KNN classifica um ponto novo com base nos vizinhos mais próximos, e que o valor de K controla diretamente o overfitting, com K=1 o modelo decora cada ponto de treino individualmente, e com K muito alto ele generaliza demais e perde detalhes importantes. A lógica faz sentido uma vez que quanto menos vizinhos você consulta, mais sensível o modelo fica a cada exemplo específico. Onde me confundo é em como escolher o K certo na prática, sei que K=1 é overfitting quase certo, mas não tenho intuição de onde está o equilíbrio.
## 8. Uso de IA, se houver
Se você usou IA depois da primeira tentativa, registre:

```text
Pergunta feita:
Resumo da resposta:
Como eu verifiquei:
O que eu alterei na minha resposta:
O que ainda não entendi:
```
Não fiz uso dos Sistemas de IA disponíveis, imagino que será usual uma das obras indicadas pelo antigo professor escrita por Aurélien Géron, intitulada Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow. 
## Submissão no Moodle

Depois de finalizar, copie no Moodle:

```text

Autoavaliação: nível atual, maior dificuldade e tópico que precisa ser retomado.
```
