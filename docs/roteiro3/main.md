# Objetivo

O objetivo desse roteiro é mostrar a implementação de um perceptron de múltiplas camadas (MLP) sem o uso de toolkits.

# Códigos

Todos os códigos utilizados para fazer esse roteiro podem ser encontrados no repositório, cujo link está na home page desse site.
Especificamente para essa entrega, o notebook usado para gerar gráficos e fazer as transformações descritas abaixo pode ser encontrado no path:

```bash
# File Location
notebooks/entrega3/ex1.ipynb
```

# Cálculos manuais dos passos de um MLP

* **Vetores de Input e Output**
  x =  [0.5, -0.2]
  y = 1.0

* **Pesos da camada oculta**
  W1 = [[0.3, -0.1], [0.2, 0.4]]

* **Bias da camada oculta**
  b1 = [0.1, -0.2]

* **Pesos da camada de saída**
  W2 = [0.5, -0.3]

* **Bias da camada de saída**
  b2 = 0.2

* **Taxa de aprendizagem**
  η = 0.3

* **Função de ativação**
  tanh

---

## 1. Forward Pass

### Computar os valores de pré-ativação da camada oculta

```
z1 = W1 @ x + b1

z1 = [[0.3, -0.1], [0.2, 0.4]] @ [0.5, -0.2] + [0.1, -0.2]

z1 = [0.3*0.5 + (-0.1)(-0.2), 0.2*0.5 + 0.4*(-0.2)] + [0.1, -0.2]

z1 = [0.15 + 0.02, 0.10 - 0.08] + [0.1, -0.2]

z1 = [0.27, -0.18]
```

### Aplicar tanh para conseguir ativação oculta

```
h1 = tanh(z1)

h1 = [tanh(0.27), tanh(-0.18)]

h1 ≈ [0.2637, -0.1781]
```

### Computar os valores de pré-ativação da camada de saída

```
u2 = W2 @ h1 + b2

u2 = [0.5, -0.3] @ [0.2637, -0.1781] + 0.2

u2 = (0.5*0.2637) + (-0.3*-0.1781) + 0.2

u2 = 0.1319 + 0.0534 + 0.2

u2 = 0.3853
```

### Ativação do output

```
y_pred = tanh(u2)

y_pred = tanh(0.3853)

y_pred ≈ 0.3672
```

---

## 2. Loss

```
L = 1/2 * (y - y_pred)^2

L = 1/2 * (1 - 0.3672)^2

L = 1/2 * (0.6328^2)

L = 1/2 * 0.4004

L ≈ 0.2002
```

---

## 3. Backward Pass

### Gradiente da loss em relação à saída

```
dL/dy_pred = (y_pred - y)

dL/dy_pred = (0.3672 - 1)

dL/dy_pred = -0.6328
```

### Gradiente em relação à pré-ativação de saída

```
dL/du2 = dL/dy_pred * (1 - y_pred^2)

dL/du2 = -0.6328 * (1 - 0.3672^2)

dL/du2 = -0.6328 * (1 - 0.1348)

dL/du2 = -0.6328 * 0.8652

dL/du2 ≈ -0.5475
```

### Gradientes da camada de saída

* Para pesos:

```
dL/dW2 = dL/du2 * h1^T

dL/dW2 = -0.5475 * [0.2637, -0.1781]

dL/dW2 = [-0.1444, 0.0975]
```

* Para bias:

```
dL/db2 = dL/du2

dL/db2 = -0.5475
```

### Gradiente propagado para a hidden

```
dL/dh1 = W2^T * dL/du2

dL/dh1 = [0.5, -0.3]^T * -0.5475

dL/dh1 = [-0.2737, 0.1643]
```

```
dL/dz1 = dL/dh1 * (1 - h1^2)

= [-0.2737, 0.1643] ⊙ [1 - 0.2637^2, 1 - (-0.1781)^2]

= [-0.2737, 0.1643] ⊙ [0.9305, 0.9683]

= [-0.2546, 0.1590]
```

### Gradientes da camada oculta

* Para pesos:

```
dL/dW1 = x^T * dL/dz1

= [0.5, -0.2]^T * [-0.2546, 0.1590]

= [[0.5*(-0.2546), 0.5*(0.1590)],
   [-0.2*(-0.2546), -0.2*(0.1590)]]

= [[-0.1273, 0.0795],
   [ 0.0509, -0.0318]]
```

* Para bias:

```
dL/db1 = dL/dz1

= [-0.2546, 0.1590]
```

---

## 4. Atualização dos parâmetros (η = 0.1)

* Saída:

```
W2_new = W2 - η * dL/dW2

= [0.5, -0.3] - 0.1*[-0.1444, 0.0975]

= [0.5144, -0.3098]
```

```
b2_new = b2 - η * dL/db2

= 0.2 - 0.1*(-0.5475)

= 0.2548
```

* Oculta:

```
W1_new = W1 - η * dL/dW1

= [[0.3, -0.1], [0.2, 0.4]] - 0.1*[[-0.1273, 0.0795], [0.0509, -0.0318]]

= [[0.3127, -0.1079],
   [0.1950,  0.4032]]
```

```
b1_new = b1 - η * dL/db1

= [0.1, -0.2] - 0.1*[-0.2546, 0.1590]

= [0.1255, -0.2159]
```

---

## Classificação binária com dados sintéticos e MLP

### Gerando os dados

Os dados utilizados foram feitos usando a função `make_classification` do scikit-learn.
O objetivo foi gerar 1000 amostras, igualmente divididas entre 2 classes, uma das classes com 1 cluster e a outra com 2.

De modo a atingir isso, foram feitas 2 chamadas diferentes, uma com o parâmetro `n_clusters_per_class` igual a 1, enquanto na outra igual a 2.
Além disso, as amostras tinham 2 features.

Esses dados foram separados em treino (80%) e teste (20%).

### Construindo a MLP

O propósito desse primeiro MLP foi desenvolver uma rede neural mais básica, com apenas 1 camada oculta, usando a **sigmoid** como função de ativação (perfeita devido à quantidade de classes = 2).
Como função de loss optei pela **Binary Cross-Entropy**.
Como otimizador usei a descida do gradiente.

Foram testadas duas arquiteturas:

* [2, 4, 1]
* [2, 8, 1]

Pesos/bias foram inicializados aleatoriamente (`np.random.randn`).
Learning rate = 0.1
Treinamento = 500 epochs.

### Resultados

* Com 4 neurônios na oculta:
  Acurácia = 0.67

  ![Variação do Loss](./loss_4.png)
  *Variação do Loss ao decorrer do treino*

  ![Decision Boundary](./boundary_4.png)
  *Decision Boundary que o modelo chegou*

* Com 8 neurônios na oculta:
  Acurácia = 0.715

  ![Variação do Loss](./loss_8.png)
  *Variação do Loss ao decorrer do treino*

  ![Decision Boundary](./boundary_8.png)
  *Decision Boundary que o modelo chegou*

---

## Classificação multiclasse com dados sintéticos e MLP

### Gerando os dados

Os dados foram feitos com `make_classification` (1500 amostras), igualmente divididos entre 3 classes:

* primeira classe com 2 clusters,
* segunda com 3 clusters,
* terceira com 4 clusters.

As amostras tinham 4 features.
Treino/teste = 80/20.

### Construindo a MLP

A implementação anterior foi adaptada para suportar múltiplas classes.
Foi adicionado o parâmetro `loss_type`, que define se será usada **BCE** (binário) ou **CCE** (multiclasse).

* Labels `y` passaram por **one-hot encoding**.
* No caso multiclasse, a saída usa **softmax**.
* Função de loss = **Categorical Cross-Entropy**.

Arquitetura testada:

* [4, 16, 3]
* 500 epochs
* lr = 0.1

### Resultados

* Acurácia = 0.546

  ![Variação do Loss](./loss_multi.png)
  *Variação do Loss ao decorrer do treino*

  ![Matriz de Confusão](./matriz_multi.png)
  *Matriz de confusão do modelo*

> Obs: O plot de *decision boundary* não faz sentido em 4D, por isso optei por mostrar a matriz de confusão.
