from torchvision import transforms

# Transformation pipeline for CIFAR-10 images
CIFAR10_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # transforms.Normalize(
        #     (0.5, 0.5, 0.5),
        #     (0.5, 0.5, 0.5),
        # ),  # Normalize the image to have mean 0 and variance 1
        transforms.Lambda(
            lambda x: x.permute(1, 2, 0)
        ),  # Change the image layout to (H, W, C)
    ]
)

# Inverse transformation for visualization
CIFAR10_INVERSE_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Change layout to (C, H, W)
        # transforms.Normalize(
        #     (-1.0, -1.0, -1.0),
        #     (2.0, 2.0, 2.0),
        # ),  # Denormalize the image
        transforms.ToPILImage(),  # Convert back to PIL Image
    ]
)

# Transformation to tensor
TENSOR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 3 else x),
    ]
)

# Transformation to PIL image
PIL_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.squeeze(0) if x.size(0) == 1 else x),
        transforms.ToPILImage(), 
    ]
)
