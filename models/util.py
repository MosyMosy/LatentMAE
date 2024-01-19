
def patchify(input_tensor, patch_size=14):
    batch_size, num_channels, height, width = input_tensor.size()

    # Calculate the number of patches in height and width
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size

    # Reshape the tensor to split it into patches
    patches = input_tensor.view(
        batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size)

    # Swap dimensions to have (batch_size, num_patches_height, num_patches_width, num_channels, patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()

    # Reshape to (batch_size * num_patches_height * num_patches_width, num_channels, patch_size, patch_size)
    patches = patches.view(-1, num_channels, patch_size, patch_size)

    return patches

def unpatchify(patches, original_size, patch_size=14):
    batch_size, num_channels, height, width = original_size

    # Calculate the number of patches in height and width
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size

    # Reshape the patches to the original tensor shape
    reshaped_tensor = patches.view(
        batch_size, num_patches_height, num_patches_width, num_channels, patch_size, patch_size)

    # Swap dimensions to have (batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    reshaped_tensor = reshaped_tensor.permute(
        0, 3, 1, 4, 2, 5).contiguous()

    # Reshape to the original tensor size
    reshaped_tensor = reshaped_tensor.view(original_size)

    return reshaped_tensor

