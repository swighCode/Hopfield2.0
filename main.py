import os
import torch
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from censor import gaussian_blur_whole_face, mosaic_whole_face, censor_eyes
from image_processing import process_image, tensor_to_image
from hopfield_network import Hopfield
from classical_hopfield import ClassicalHopfield
from torch.nn.functional import normalize
from skimage.filters import threshold_otsu

# Mapping of method names to censoring functions
CENSOR_METHODS = {
    'gaussian': gaussian_blur_whole_face,
    'eyes_censor': censor_eyes,
    'mosaic': mosaic_whole_face,
}

# Parameter ranges for each method (to sweep)
CENSOR_PARAM_SWEEPS = {
    'gaussian': [5, 20, 50],
    'eyes_censor': [None],  # No parameter for eyes_censor
    'mosaic': [5, 20, 50],
}


def load_patterns(folder: str, max_images: int = None, bipolar=False):
    images = []
    labels = []
    collected = 0
    for root, _, files in os.walk(folder):
        for file in files:
            parts = file.split('_')
            path = os.path.join(root, file)
            img = process_image(path)
            images.append(img)
            labels.append(parts[0])
            collected += 1
            if max_images and collected >= max_images:
                break
    if bipolar:
        bipolars = []
        for img in images:
            bipolar = apply_atkinson(img)
            bipolars.append(bipolar)
        patterns = torch.stack(bipolars).T
    else:
        patterns = torch.stack(images).T

    return patterns, labels
    
def apply_atkinson(img_tensor):
    # Convert the tensor to a numpy array and reshape to 2D
    img_np = img_tensor.detach().clone().numpy()
    length = img_np.size
    side = int(math.sqrt(length))
    if side * side != length:
        raise ValueError("Image tensor must be a perfect square.")
    img_2d = img_np.reshape((side, side))
    
    # Create a float copy for error diffusion
    img_float = img_2d.astype(np.float32)
    
    for y in range(side):
        for x in range(side):
            old_pixel = img_float[y, x]
            new_pixel = 1.0 if old_pixel >= 0.5 else 0.0
            img_float[y, x] = new_pixel
            error = old_pixel - new_pixel
            
            # Distribute error according to Atkinson's algorithm
            error_frac = error / 8.0
            # Define the offsets for error distribution
            offsets = [
                (1, 0),   # Right 1
                (2, 0),   # Right 2
                (-1, 1),  # Left 1, Down 1
                (0, 1),   # Down 1
                (1, 1),   # Right 1, Down 1
                (0, 2),   # Down 2
            ]
            for dx, dy in offsets:
                xn = x + dx
                yn = y + dy
                if 0 <= xn < side and 0 <= yn < side:
                    img_float[yn, xn] += error_frac
    
    # Convert to binary (0s and 1s) after dithering
    binary_img = (img_float >= 0.5).astype(np.float32)
    
    # Convert binary image to bipolar (-1 and 1)
    bipolar_img = binary_img * 2 - 1
    
    # Flatten and return as a tensor
    return torch.tensor(bipolar_img.flatten(), dtype=torch.float32)

def evaluate_retrieval(patterns: torch.Tensor, labels: list, test_folder: str, method: str, model_type="modern", param_val: int = None):

    if model_type == "modern":
        hopnet = Hopfield(patterns)
    else:
        hopnet = ClassicalHopfield(patterns)

    success = 0
    total = 0

    for root, _, files in os.walk(test_folder):
        for file in files:
            parts = file.split('_')
            label = parts[0]
            if label not in labels:
                continue


            '''
            -----------------
            IMAGE PROCESSING
            -----------------
            '''
            src_path = os.path.join(root, file)
            if method == 'original':
                query_img = process_image(src_path)
            elif method == 'eyes_censor':
                tmp_dst = os.path.join('tmp', method, file)
                os.makedirs(os.path.dirname(tmp_dst), exist_ok=True)
                censor_eyes(src_path, tmp_dst)
                query_img = process_image(tmp_dst)
            else:
                tmp_dst = os.path.join('tmp', f"{method}_{param_val}", file)
                os.makedirs(os.path.dirname(tmp_dst), exist_ok=True)
                func = CENSOR_METHODS[method]
                param_key = 'sigma' if method == 'gaussian' else 'mosaic_size'
                func(src_path, tmp_dst, **{param_key: param_val})
                query_img = process_image(tmp_dst)

            '''
            -----------------
            RETRIEVAL
            -----------------
            '''
            if model_type == "modern":
                state = query_img.clone()
                prev = torch.zeros_like(state)
                for _ in range(100):
                    prev.copy_(state)
                    state = hopnet.forward(state)
                    if torch.allclose(state, prev, atol=1e-4):
                        break

            else:
                state = apply_atkinson(query_img)
                tensor_to_image(state)
                prev = torch.zeros_like(state)
                for _ in range(100):
                    prev = state.clone()
                    state = hopnet.forward(state)
                    if torch.allclose(state, prev, atol=1e-4):
                        break
                state[state == 0] = 1  # Handle any remaining zeros
                tensor_to_image(state)

            '''
            -----------------
            MATCHING
            -----------------
            '''

            sims = patterns.T @ state
            
            if model_type == "classical":
                # For classical network, normalize similarities by pattern size
                sims /= patterns.shape[0]

            retrieved_idx = torch.argmax(sims).item()



            retrieved = labels[retrieved_idx]
            total += 1
            if label == retrieved:
                success += 1
                print(f"Success: {label} -> {retrieved}")
            else:
                print(f"Failure: {label} vs {retrieved}")

    return success / total if total > 0 else 0.0


def main():
    training_folder = 'dataset/train'
    test_folder = 'dataset/test'
    image_counts = list(range(5, 20, 5))
    results = []

    for model_type in ['modern', 'classical']:
        for count in image_counts:
            patterns, labels = load_patterns(training_folder, max_images=count, bipolar=(model_type == "classical"))
            for method in ['original']:
                print(f"Evaluating {method} with {count} images...")
                acc = evaluate_retrieval(patterns, labels, test_folder, method, model_type=model_type)
                results.append({
                    'num_images': count,
                    'method': method,
                    'param': 'none',
                    'accuracy': acc,
                })

            for method in CENSOR_METHODS:
                for param in CENSOR_PARAM_SWEEPS[method]:
                    method_label = f"{method}_{param}" if param is not None else method
                    print(f"Evaluating {method_label} with {count} images...")
                    acc = evaluate_retrieval(patterns, labels, test_folder, method, param_val=param, model_type=model_type)
                    results.append({
                        'num_images': count,
                        'method': method_label,
                        'param': param if param is not None else 'none',
                        'model_type': model_type,
                        'accuracy': acc,
                    })

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    pivot_df = df.pivot_table(index='method', columns='num_images', values='accuracy')
    pivot_df.to_csv('results_pivot.csv')
    print("\nRetrieval Rates Table:")
    print(df.pivot_table(index='method', columns='num_images', values='accuracy'))


    # Define consistent styling
    METHOD_STYLES = {
        'gaussian': {'color': '#1f77b4', 'marker': 'o'},  # Blue
        'eyes_censor': {'color': '#2ca02c', 'marker': 's'},  # Green
        'mosaic': {'color': '#d62728', 'marker': 'D'},  # Red
    }

    MODEL_STYLES = {
        'modern': {'linestyle': '-', 'linewidth': 2},
        'classical': {'linestyle': '--', 'linewidth': 1.5},
    }

    # Create subplots for each censoring type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    censor_types = ['gaussian', 'eyes_censor', 'mosaic']

    for ax, censor_type in zip(axes, censor_types):
        # Filter data for current censoring type
        subset = df[df['method'].str.contains(censor_type) | (df['method'] == 'original')]
        
        # Split modern vs classical
        for model_type in ['modern', 'classical']:
            model_subset = subset[subset['model_type'] == model_type]
            
            # Plot each parameter variation
            for method in model_subset['method'].unique():
                method_data = model_subset[model_subset['method'] == method]
                param = method.split('_')[-1] if '_' in method else 'none'
                
                # Style mapping
                style = {
                    **METHOD_STYLES[censor_type],
                    **MODEL_STYLES[model_type],
                    'markersize': 8,
                    'alpha': 0.8 if model_type == 'modern' else 0.6
                }
                
                ax.plot(
                    method_data['num_images'],
                    method_data['accuracy'],
                    label=f"{model_type.capitalize()} ({param})" if param != 'none' else f"{model_type.capitalize()}",
                    **style
                )

        ax.set_xlabel('Number of Training Images')
        ax.set_title(censor_type.capitalize().replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)  # Consistent Y-axis

    # Global labels and legend
    axes[0].set_ylabel('Retrieval Accuracy')
    fig.suptitle('Modern vs classical Hopfield Network Performance by Censoring Type', y=1.05, fontsize=14)

    # Create unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
            loc='upper center', 
            ncol=4, 
            bbox_to_anchor=(0.5, 0.95),
            frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
