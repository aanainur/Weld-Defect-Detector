import numpy as np
import cv2
import os
from scipy.signal import convolve2d

# ===================== CONFIG =====================
BASE_DIR = r"c:\Users\aanai\OneDrive\Documents\Skripsi\dataseet\dataset cnn 4 class"
MODEL_PATH = os.path.join(BASE_DIR, "cnn_manual_448x224.npz")

# ===================== ACTIVATION =====================
def relu(x): 
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ===================== POOLING =====================
def max_pool(img, size=2):
    h, w = img.shape
    out_h = h // size
    out_w = w // size
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i*size:(i+1)*size, j*size:(j+1)*size]
            out[i, j] = np.max(patch)
    return out

# ===================== LOAD MODEL =====================
def load_model(model_path):
    """Load trained model from npz file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    data = np.load(model_path)
    
    model = {
        'kernels': data['kernels'],
        'W1': data['W1'],
        'b1': data['b1'],
        'W2': data['W2'],
        'b2': data['b2'],
        'IMG_W': int(data['IMG_W']),
        'IMG_H': int(data['IMG_H']),
        'class_names': list(data['class_names'])
    }
    
    print(f"[OK] Model loaded from {model_path}")
    print(f"     Image size: {model['IMG_W']}x{model['IMG_H']}")
    print(f"     Classes: {', '.join(model['class_names'])}")
    
    return model

# ===================== PREPROCESSING =====================
def preprocess_image(img_path, img_w, img_h):
    """Load and preprocess image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Resize
    img = cv2.resize(img, (img_w, img_h))
    
    # Denoising dengan Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Normalisasi 0-1
    img = img / 255.0
    
    return img

# ===================== INFERENCE =====================
def predict(model, img_path):
    """Predict class for given image"""
    
    # Preprocess
    img = preprocess_image(img_path, model['IMG_W'], model['IMG_H'])
    
    kernels = model['kernels']
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    
    # Forward pass
    maps = []
    for k in kernels:
        c = convolve2d(img, k, mode="valid")
        a = relu(c)
        p = max_pool(a)
        maps.append(p)
    
    feat = np.array(maps).flatten()
    h = relu(feat @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    
    pred_idx = np.argmax(probs)
    pred_class = model['class_names'][pred_idx]
    confidence = probs[pred_idx] * 100
    
    return {
        'class': pred_class,
        'confidence': confidence,
        'class_idx': pred_idx,
        'all_probs': probs,
        'all_classes': model['class_names']
    }

# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("CNN MODEL INFERENCE")
    print("=" * 60)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    print("\n[INFO] Prediksi Mode:")
    print("  1. Prediksi satu gambar")
    print("  2. Prediksi folder validation")
    print("  3. Prediksi folder train")
    print("  4. Custom folder")
    
    choice = input("\nPilih mode (1-4): ").strip()
    
    if choice == "1":
        # Single image
        img_path = input("Masukkan path file gambar: ").strip()
        result = predict(model, img_path)
        
        print(f"\n[RESULT] Prediksi: {result['class']} (Confidence: {result['confidence']:.2f}%)")
        print("\nDetail semua kelas:")
        for cls, prob in zip(result['all_classes'], result['all_probs']):
            print(f"  {cls}: {prob*100:.2f}%")
    
    elif choice == "2" or choice == "3" or choice == "4":
        # Folder mode
        if choice == "2":
            folder_path = os.path.join(BASE_DIR, "valid")
        elif choice == "3":
            folder_path = os.path.join(BASE_DIR, "train")
        else:
            folder_path = input("Masukkan path folder: ").strip()
        
        # Collect true labels from subfolder structure
        results = []
        total, correct = 0, 0
        
        classes = sorted(os.listdir(folder_path))
        for true_label, cls_name in enumerate(classes):
            cls_path = os.path.join(folder_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            
            for img_file in os.listdir(cls_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                
                img_path = os.path.join(cls_path, img_file)
                pred = predict(model, img_path)
                
                is_correct = (pred['class_idx'] == true_label)
                results.append({
                    'file': img_file,
                    'true_class': cls_name,
                    'pred_class': pred['class'],
                    'confidence': pred['confidence'],
                    'correct': is_correct
                })
                
                total += 1
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                print(f"{status} {img_file} → {pred['class']} ({pred['confidence']:.1f}%)")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\n[SUMMARY] Accuracy: {correct}/{total} = {accuracy:.2f}%")
        
        # Confusion matrix summary
        print("\nConfusion per class:")
        for cls_name in model['class_names']:
            class_results = [r for r in results if r['true_class'] == cls_name]
            if class_results:
                cls_acc = sum(1 for r in class_results if r['correct']) / len(class_results) * 100
                print(f"  {cls_name}: {cls_acc:.1f}% ({sum(1 for r in class_results if r['correct'])}/{len(class_results)})")

if __name__ == "__main__":
    main()
