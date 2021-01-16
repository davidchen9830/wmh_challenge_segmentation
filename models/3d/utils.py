import numpy as np

def pad(image, input_size, output_size):
    _, height, width, _ = image.shape
    
    wanted_height = (height - 1) // output_size * output_size + output_size
    wanted_width = (width - 1) // output_size * output_size + output_size
    
    top_padding = left_padding = (input_size - output_size) // 2
    bottom_padding = wanted_height - height + (input_size - output_size) // 2
    right_padding = wanted_width - width + (input_size - output_size) // 2
    
    return np.pad(image, ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding), (0, 0)), mode='symmetric')

def get_slices(image, input_size, output_size):
    _, height, width, _ = image.shape
    
    top_padding = bottom_padding = left_padding = right_padding = (input_size - output_size) // 2
    
    height -= top_padding + bottom_padding
    width -= left_padding + right_padding
    
    images = []
    y = top_padding
    while y < height:
        x = left_padding
        while x < width:
            subim = image[:, y-top_padding:y+output_size+bottom_padding, x-left_padding:x+output_size+right_padding, :]
            images.append(subim)
            x += output_size
        y += output_size
    return np.swapaxes(images, 0, 1)

def reconstruct(slices, shape, output_size=None):
    assert(slices.shape[2] == slices.shape[3])
    input_size = slices.shape[2]
    if output_size is None:
        output_size = input_size
    
    n = slices.shape[0]
    channels = slices.shape[-1]
    height, width = shape
    nb_y = (height - 1) // output_size + 1
    nb_x = (width - 1) // output_size + 1
    wanted_height = nb_y * output_size
    wanted_width = nb_x * output_size
    
    top_padding = bottom_padding = left_padding = right_padding = (input_size - output_size) // 2
    
    output = np.zeros((n, wanted_height, wanted_width, channels), dtype=slices.dtype)
    for yi in range(nb_y):
        y = yi * output_size
        for xi in range(nb_x):
            x = xi * output_size
            output[:, y:y+output_size, x:x+output_size] = slices[:, yi * 3 + xi, top_padding:input_size-bottom_padding, left_padding:input_size-right_padding]
    return output[:, :height, :width, :]