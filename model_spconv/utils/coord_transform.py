import torch

def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.atan2(input_xyz[:, 0], input_xyz[:, 1])
    return torch.stack((rho, phi, input_xyz[:, 2]), dim=1)