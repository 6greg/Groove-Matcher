import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr


# caricamento tensori
def load_tensor(path):
    """carica un tensore e restituisce X, Y, ids e nomi dei canali"""
    tensor = np.load(path)
    return dict(X=tensor["X"].astype(float),
                Y=tensor["Y"].astype(float),
                ids=tensor["ids"],)


# preprocessing
def prepare(tensor):
    """centra ogni canale di ogni traccia sottraendo la sua media nel tempo"""
    return tensor - tensor.mean(axis=2, keepdims=True)

def pool(tensor):
    """da (n, f, s) a (n·s, f): una riga per ogni (traccia, step), una colonna per canale"""
    n, f, s = tensor.shape
    return tensor.transpose(0, 2, 1).reshape(n * s, f)

# modello
def fit(X, Y):
    """preprocessing + CCA a una componente sulle righe pooled"""
    Xc = prepare(X) 
    Yc = prepare(Y)
    cca = CCA(n_components=1, scale=True, max_iter=1000)
    cca.fit(pool(Xc), pool(Yc))   # trova i pesi di X e di Y
    return dict(cca=cca)

# pearson correlation
def pearson_corr(model, X, Y):
    """correlazione canonica tra le prime variate canoniche"""
    Xc = prepare(X)
    Yc = prepare(Y)
    u, v = model["cca"].transform(pool(Xc), pool(Yc))   # variate canoniche (u per gli arrangiamenti, v per le batterie)
    return pearsonr(u[:, 0], v[:, 0]).statistic

