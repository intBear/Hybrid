import torch
from utils import *

def to_CordsAndValues(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    values = img.reshape(img.shape[1], img.shape[2], -1)
    return coordinates, values

if __name__ == "__main__":
    a = torch.randn(3, 8, 8)
    cords, values = to_CordsAndValues(a)
    # cords = cords.reshape(8, 8, 2)
    encoder, out_dim = get_positional_embedder(4, True)
    encoded = encoder(cords)
    encoded = encoded.reshape(8, 8, out_dim)
    z = torch.randn(8, 8, 7)
    z = torch.cat((z, encoded), 2)
    print(z.shape)