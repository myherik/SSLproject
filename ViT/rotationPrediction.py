import torchvision
import torch
from torchvision import transforms

    
class RotationPredictionDataset(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, **kwargs):
        super(RotationPredictionDataset, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        image, _ = super(RotationPredictionDataset, self).__getitem__(index)
        
        # Rotate the image by a random angle (0, 90, 180, or 270 degrees)
        angle = torch.randint(0, 4, size=(1,)).item()  # 0, 1, 2, or 3
        rotated_image = transforms.functional.rotate(image, angle * 90)
        
        return rotated_image, angle