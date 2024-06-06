import matplotlib.pyplot as plt
import numpy as np

data = {"Feature1": [1.0, 2.0, 1.5, 3.0, 3.5], "Feature2": [1.0, 1.0, 2.0, 3.5, 3.0]}

plt.scatter(data["Feature1"], data["Feature2"], color="blue")

plt.title("Gráfico de Dispersão das Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")


def inicializar_centroídes(X, K):
    """Inicializa K centróides aleatórios a partir do conjunto de dados X"""
    centroídes = X.copy()
    np.random.shuffle(centroídes)
    return centroídes[:K]


def atribuir_clusters(X, centroídes):
    """Atribui cada ponto de X ao cluster do centróide mais próximo"""
    distâncias = np.sqrt(((X - centroídes[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distâncias, axis=0)


def atualizar_centroídes(X, labels, K):
    """Recalcula os centróides como a média de todos os pontos em cada cluster"""
    novos_centroídes = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    return novos_centroídes


def kmeans(X, K, max_iters=100, tol=1e-4):
    """Implementação do algoritmo K-means"""
    centroídes = inicializar_centroídes(X, K)
    for _ in range(max_iters):
        labels = atribuir_clusters(X, centroídes)
        novos_centroídes = atualizar_centroídes(X, labels, K)
        if np.all(np.abs(novos_centroídes - centroídes) < tol):
            break
        centroídes = novos_centroídes
    return centroídes, labels


X = np.array(list(zip(data["Feature1"], data["Feature2"])))
K = 2

centroídes, labels = kmeans(X, K)

plt.scatter(data["Feature1"], data["Feature2"], c=labels, cmap="viridis")
plt.scatter(centroídes[:, 0], centroídes[:, 1], c="red", s=100, alpha=0.5)

plt.show()
