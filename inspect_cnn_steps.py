import argparse
import os
import numpy as np
import cv2
from scipy.signal import convolve2d
import tkinter as tk
from tkinter import filedialog

# File model default yang digunakan jika tidak ada argumen model
DEFAULT_MODEL = "cnn_manual_124x124_001_76persen.npz"


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def max_pool(img, size=2):
    h, w = img.shape
    out_h = h // size
    out_w = w // size
    out = np.zeros((out_h, out_w), dtype=img.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i*size:(i+1)*size, j*size:(j+1)*size]
            out[i, j] = np.max(patch)
    return out


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data = np.load(model_path, allow_pickle=True)
    model = {
        "kernels": data["kernels"],
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
        "IMG_W": int(data["IMG_W"]),
        "IMG_H": int(data["IMG_H"]),
        "class_names": list(data["class_names"])
    }

    return model


def preprocess_image(img_path, img_w, img_h):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.resize(img, (img_w, img_h))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float32) / 255.0
    return img


def format_matrix(matrix, max_rows=6, max_cols=6, precision=5):
    if matrix.ndim == 1:
        if matrix.size <= max_rows * 2:
            return np.array2string(matrix, precision=precision, suppress_small=True)
        return np.array2string(matrix[:max_rows], precision=precision, suppress_small=True) + " ..."

    h, w = matrix.shape
    if h * w <= max_rows * max_cols:
        return np.array2string(matrix, precision=precision, suppress_small=True)

    preview = matrix[:max_rows, :max_cols]
    text = np.array2string(preview, precision=precision, suppress_small=True)
    return text.replace("\n", "\n") + f"\n... (shape={h}x{w})"


def print_matrix(name, matrix, limit_print=True):
    # Cetak informasi bentuk dan isi matriks agar lebih mudah dipahami
    print(f"\n=== {name} ===")
    print(f"shape = {matrix.shape}, dtype = {matrix.dtype}")
    if isinstance(matrix, np.ndarray):
        if limit_print and matrix.size > 200:
            print(format_matrix(matrix))
            print(f"(matrix too large to print fully; showing top-left corner)")
        else:
            print(np.array2string(matrix, precision=5, suppress_small=True))
    else:
        print(matrix)


def inspect_cnn_steps(model, img_path, output_dir=None):
    # Muat dan proses citra input dengan ukuran model
    img = preprocess_image(img_path, model["IMG_W"], model["IMG_H"])
    print(f"Loaded image: {img_path}")
    print(f"Image size after preprocessing: {img.shape}")
    print_matrix("Input image", img)

    kernels = model["kernels"]
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    saved = {}
    feature_maps = []

    for idx, kernel in enumerate(kernels):
        print(f"\n--- Kernel {idx + 1}/{len(kernels)} ---")
        print_matrix(f"Kernel #{idx}", kernel)

        # Konvolusi antara citra dan kernel
        conv_map = convolve2d(img, kernel, mode="valid")
        print_matrix(f"Conv result #{idx}", conv_map)

        # Aktivasi ReLU setelah konvolusi
        relu_map = relu(conv_map)
        print_matrix(f"ReLU output #{idx}", relu_map)

        # Max pooling untuk mengurangi dimensi fitur
        pool_map = max_pool(relu_map)
        print_matrix(f"Max pooled output #{idx}", pool_map)

        feature_maps.append(pool_map)
        saved[f"kernel_{idx}"] = kernel
        saved[f"conv_{idx}"] = conv_map
        saved[f"relu_{idx}"] = relu_map
        saved[f"pool_{idx}"] = pool_map

    feat = np.array(feature_maps).flatten()
    print_matrix("Flattened feature vector", feat)
    saved["flattened_features"] = feat

    hidden = relu(feat @ W1 + b1)
    print_matrix("Hidden layer activation", hidden)
    saved["hidden_activation"] = hidden

    logits = hidden @ W2 + b2
    print_matrix("Logits (pre-softmax)", logits)
    saved["logits"] = logits

    probabilities = softmax(logits)
    print_matrix("Softmax probabilities", probabilities)
    saved["probabilities"] = probabilities

    top_idx = np.argsort(probabilities)[::-1]
    print("\n=== Prediction summary ===")
    for rank, idx in enumerate(top_idx[:5], start=1):
        label = model["class_names"][idx]
        score = probabilities[idx] * 100
        print(f"{rank}. {label}: {score:.2f}%")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"inspection_{base_name}.npz")
        np.savez_compressed(save_path, **saved)
        print(f"\nSaved all intermediate matrices to: {save_path}")


def select_image_file(initial_dir=None):
    # Buka dialog file untuk memilih citra secara manual
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select input image",
        initialdir=initial_dir or os.getcwd(),
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*")
        ]
    )
    root.destroy()
    return file_path


def parse_args():
    # Parsir argumen command line yang digunakan untuk menjalankan skrip
    parser = argparse.ArgumentParser(description="Inspect CNN matrix values step-by-step")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to CNN model .npz file")
    parser.add_argument("--output", default="inspection_output", help="Folder to save intermediate matrices")
    parser.add_argument("--no-save", action="store_true", help="Do not save intermediate matrices to disk")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image:
        img_path = os.path.abspath(args.image)
    else:
        print("No image passed as argument. Opening file dialog...")
        img_path = select_image_file()
        if not img_path:
            print("No image selected. Exiting.")
            return

    model_path = os.path.abspath(args.model)
    output_dir = None if args.no_save else os.path.abspath(args.output)

    # Muat model dan jalankan inspeksi langkah demi langkah
    model = load_model(model_path)
    inspect_cnn_steps(model, img_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
