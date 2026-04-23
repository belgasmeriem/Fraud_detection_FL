import numpy as np

"""
Differential Privacy — Privacy Engine
======================================
Formule exacte du prof (PDF pages 3-4) :
    1. Clipping L2 : delta_W_clip = delta_W / max(1, ||delta_W||_2 / C)
    2. Bruit Gaussien : delta_W_private = delta_W_clip + N(0, sigma^2 * C^2)

Parametres :
    C     = 1.0   : seuil de clipping (sensitivity bound)
    sigma = 0.1   : calibre pour epsilon ~ 3.0 (budget privacy finance)

Budget privacy Finance (fichiers prof) : epsilon entre 1.0 et 5.0
Les features financieres sont deja anonymisees (PCA) -> sigma moins strict
qu'en sante (epsilon < 1.0).
"""


def clip_gradients(gradients: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    L2 Clipping — borne la sensibilite du gradient.
    Empeche un noeud malveillant d'envoyer des gradients arbitrairement grands.
    Formule : delta_W_clip = delta_W / max(1, ||delta_W||_2 / C)
    """
    norm = np.linalg.norm(gradients)
    if norm > C:
        return gradients * (C / norm)
    return gradients


def add_gaussian_noise(gradients: np.ndarray, C: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """
    Bruit Gaussien calibre au budget privacy.
    Mecanisme de Gauss : bruit proportionnel a N(0, sigma^2 * C^2 * I)
    """
    noise = np.random.normal(0, sigma * C, size=gradients.shape)
    return gradients + noise


def apply_dp(gradients: np.ndarray, C: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """
    Pipeline DP complet applique sur les poids PyTorch avant envoi au serveur.

    IMPORTANT : applique sur les VRAIS poids envoyes (get_params),
    pas sur les predictions intermediaires.
    Reste en float32 — coherent avec les tenseurs PyTorch.

    Args:
        gradients : numpy array des poids du modele local (float32)
        C         : seuil de clipping L2 (sensitivity)
        sigma     : niveau de bruit (epsilon ~ 3.0 pour sigma=0.1)

    Returns:
        numpy array avec DP appliquee, meme dtype que l'entree
    """
    p = gradients.astype(np.float32)
    p = clip_gradients(p, C)
    p = add_gaussian_noise(p, C, sigma)
    return p
