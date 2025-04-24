import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from censor import gaussian_blur_whole_face, mosaic_whole_face, censor_eyes
from image_processing import process_image, tensor_to_image
from hopfield_network import Hopfield

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


def load_patterns(folder: str, max_images: int = None):
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
        if max_images and collected >= max_images:
            break
    patterns = torch.stack(images).T
    return patterns, labels


def evaluate_retrieval(patterns: torch.Tensor, labels: list, test_folder: str, method: str, param_val: int = None):
    hopnet = Hopfield(patterns)
    success = 0
    total = 0

    for root, _, files in os.walk(test_folder):
        for file in files:
            parts = file.split('_')
            label = parts[0]
            if label not in labels:
                continue

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

            state = query_img.clone()
            prev = torch.zeros_like(state)
            for _ in range(100):
                prev.copy_(state)
                state = hopnet.forward(state)
                if torch.allclose(state, prev, atol=1e-4):
                    break

            sims = torch.matmul(patterns.T, state)
            retrieved = labels[torch.argmax(sims)]
            total += 1
            if retrieved == label:
                success += 1
    print(total)
    return success / total if total > 0 else 0.0


def main():
    training_folder = 'dataset/train'
    test_folder = 'dataset/test'
    image_counts = [5, 10, 20, 40, 80, 100, 150, 186]
    results = []

    for count in image_counts:
        patterns, labels = load_patterns(training_folder, max_images=count)
        for method in ['original']:
            acc = evaluate_retrieval(patterns, labels, test_folder, method)
            results.append({
                'num_images': count,
                'method': method,
                'param': 'none',
                'accuracy': acc,
            })
            print(f"Count: {count}, Method: {method}, Accuracy: {acc:.2%}")

        for method in CENSOR_METHODS:
            for param in CENSOR_PARAM_SWEEPS[method]:
                method_label = f"{method}_{param}" if param is not None else method
                acc = evaluate_retrieval(patterns, labels, test_folder, method, param_val=param)
                results.append({
                    'num_images': count,
                    'method': method_label,
                    'param': param if param is not None else 'none',
                    'accuracy': acc,
                })
                print(f"Count: {count}, Method: {method}, Param: {param}, Accuracy: {acc:.2%}")

    df = pd.DataFrame(results)
    print("\nRetrieval Rates Table:")
    print(df.pivot_table(index='method', columns='num_images', values='accuracy'))

    plt.figure()
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        plt.plot(
            subset['num_images'],
            subset['accuracy'],
            marker='o',
            label=method
        )

    plt.xlabel('Number of Training Images')
    plt.ylabel('Retrieval Accuracy')
    plt.title('Hopfield Retrieval Accuracy by Censoring Method & Amount')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
