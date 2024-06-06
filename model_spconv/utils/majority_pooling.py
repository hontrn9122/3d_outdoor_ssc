import torch

def majority_pooling(input_tensor, kernel_size):
    assert input_tensor.dim() in [4, 5]
    
    if input_tensor.dim() == 5: 
        return batch_majority_pooling(input_tensor, kernel_size)
    
    dim, height, width, depth = input_tensor.size()
    output_height = height // kernel_size
    output_width = width // kernel_size
    output_depth = depth // kernel_size
    
    # Reshape input tensor for easy manipulation
    reshaped_tensor = input_tensor.view(dim, output_height, kernel_size, output_width, kernel_size, output_depth, kernel_size)
    reshaped_tensor = reshaped_tensor.permute(0,1,3,5,2,4,6).reshape(dim, output_height, output_width, output_depth, -1)
    
    return torch.mode(reshaped_tensor, dim=4).values

def batch_majority_pooling(input_tensor, kernel_size):
    batch, dim, height, width, depth = input_tensor.size()
    output_height = height // kernel_size
    output_width = width // kernel_size
    output_depth = depth // kernel_size
    
    # Reshape input tensor for easy manipulation
    reshaped_tensor = input_tensor.view(batch, dim, output_height, kernel_size, output_width, kernel_size, output_depth, kernel_size)
    reshaped_tensor = reshaped_tensor.permute(0,1,2,4,6,3,5,7).reshape(batch, dim, output_height, output_width, output_depth, -1)
    
    return torch.mode(reshaped_tensor, dim=5).values


def batch_bev_majority_pooling(input_tensor, kernel_size):
    batch, dim, height, width, depth = input_tensor.size()
    output_height = height // kernel_size
    output_width = width // kernel_size
    
    # Reshape input tensor for easy manipulation
    reshaped_tensor = input_tensor.view(batch, dim, output_height, kernel_size, output_width, kernel_size, depth)
    reshaped_tensor = reshaped_tensor.permute(0,1,2,4,3,5,6).reshape(batch, dim, output_height, output_width, -1)
    
    return torch.mode(reshaped_tensor, dim=4).values