import torch.nn as nn

def get_loss(loss_type="bce"):
    """
    Retourne la fonction de loss à utiliser pour l'entraînement.

    Args:
        loss_type (str): Type de loss. Options :
            - "bce" : Binary Cross Entropy (classification binaire)
            - "mse" : Mean Squared Error (régression)
    
    Returns:
        torch.nn.Module: instance de la loss
    """
    loss_type = loss_type.lower()
    
    if loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss non reconnue : {loss_type}. Choisir 'bce' ou 'mse'.")
