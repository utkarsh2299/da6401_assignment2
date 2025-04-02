def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)