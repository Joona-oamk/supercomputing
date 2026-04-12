# %%
# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?', default='super_data')
    parser.add_argument('output_path', nargs='?', default='rendered_png')
    return parser.parse_known_args()[0]


args = parse_args()
data_path = Path(args.input_path)
output_path = Path(args.output_path)
files = sorted(data_path.glob('*.npy'))
print(f'Number of files: {len(files)}')


# %%

data = {}
for i in range(10):
    index = np.random.randint(len(files))
    data[i] = np.load(files[index])
    print(f'{i}: {files[index].name} shape = {data[i].shape}')


sample = np.load(files[0])
print('\nSample file:', files[0].name)
print('shape =', sample.shape)
print('dtype =', sample.dtype)
print('ndim =', sample.ndim)
print('min/max =', np.min(sample), np.max(sample))
print('mean/std =', float(np.mean(sample)), float(np.std(sample)))
print('first 10 values =', sample.ravel()[:10])

# %%
# Add value of each 
total_sum = 0
for i in range(len(data)):
    value = data[i]
    total_sum += value
    

print('Total sum file')
print('shape =', total_sum.shape)
print('dtype =', total_sum.dtype)
print('ndim =', total_sum.ndim)
print('min/max =', np.min(total_sum), np.max(total_sum))
print('mean/std =', float(np.mean(total_sum)), float(np.std(total_sum)))
print('first 10 values =', total_sum.ravel()[:10])


# %%
def render_array_to_png(array, image_name, output_dir='rendered_png'):
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D array, got shape {array.shape}')

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    png_path = output_path / f'{image_name}.png'

    image = np.clip(array, 0, 255).astype(np.uint8)
    plt.imsave(png_path, image, cmap='gray', vmin=0, vmax=255)
    return png_path

for i in range(len(data)):
    print(f'\nRendering data[{i}] to PNG...')
    rendered_sample = render_array_to_png(data[i], f'data_{i}', output_dir=output_path)
    print('saved png =', rendered_sample)

plt.figure(figsize=(6, 6))
plt.imshow(sample, cmap='gray', vmin=0, vmax=255)
plt.title(files[0].name)
plt.axis('off')
plt.show()


