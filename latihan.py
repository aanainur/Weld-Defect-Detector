import numpy as np
import cv2
import os
from scipy.signal import convolve2d, correlate2d

# ===================== KONFIGURASI =====================
# Base directory untuk file ini, digunakan agar path relatif dapat bekerja
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ukuran input citra yang akan diproses oleh CNN manual
IMG_W, IMG_H = 224, 224               # RESIZE (sesuai permintaan)
KERNEL_SIZE = 3
POOL_SIZE = 2

# Jumlah filter pada layer konvolusi, ukuran hidden layer, dan jumlah kelas
NUM_KERNEL = 8
HIDDEN = 128
NUM_CLASS = 4                          # excessive, normal, porosity, undercut

# Hyperparameter training
LR = 0.01
EPOCHS = 20
BATCH = 16

# ===================== AKTIVASI =====================
# Fungsi ReLU untuk aktivasi non-linear
def relu(x): return np.maximum(0, x)
# Turunan ReLU untuk backpropagation
def relu_deriv(x): return (x > 0).astype(float)

# Fungsi softmax untuk mengubah logit menjadi probabilitas
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ===================== POOLING =====================
# Max pooling sederhana untuk mengurangi dimensi peta fitur
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

# ===================== PEMUATAN DATASET DARI FOLDER =====================
# Fungsi untuk membaca folder dataset dan mengembalikan citra serta label
def load_folder(path):
    classes = sorted(os.listdir(path))
    X, y = [], []

    for label, cls in enumerate(classes):
        cls_path = os.path.join(path, cls)
        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            img_path = os.path.join(cls_path, f)
            # Membaca citra dalam mode grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None: 
                continue

            # Resize citra agar konsisten
            img = cv2.resize(img, (IMG_W, IMG_H))
            
            # Denoising dengan Gaussian Blur untuk mengurangi noise
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Normalisasi piksel ke rentang 0-1
            img = img / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), classes

print("[INFO] Loading dataset...")
X_train, y_train, class_names = load_folder(os.path.join(BASE_DIR, "train"))
X_val, y_val, _ = load_folder(os.path.join(BASE_DIR, "valid"))

print(f"[OK] Train: {len(X_train)} | Valid: {len(X_val)}")
print("Classes:", class_names)

# ===================== INITIAL PARAMS =====================
def he(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[1])

# initialize kernels with small random values (He-like scaling)
kernels = np.random.randn(NUM_KERNEL, KERNEL_SIZE, KERNEL_SIZE) * np.sqrt(2.0 / (KERNEL_SIZE*KERNEL_SIZE))

# hitung size output conv + pool
conv_h = IMG_H - KERNEL_SIZE + 1
conv_w = IMG_W - KERNEL_SIZE + 1

pool_h = conv_h // POOL_SIZE
pool_w = conv_w // POOL_SIZE

feat_size = pool_h * pool_w * NUM_KERNEL

W1 = he((feat_size, HIDDEN))
b1 = np.zeros(HIDDEN)

W2 = he((HIDDEN, NUM_CLASS)) * 0.1
b2 = np.zeros(NUM_CLASS)

# separate learning rate for kernels (may need tuning)
LR_K = LR * 2.0

# ===================== TRAINING =====================
# Loop utama training dengan batching dan backpropagation manual
for epoch in range(EPOCHS):
    # Acak urutan data setiap epoch untuk generalisasi lebih baik
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    total_loss, correct = 0, 0

    for i in range(0, len(X_train), BATCH):
        xb = X_train[i:i+BATCH]
        yb = y_train[i:i+BATCH]

        feats = []
        cache = []

        for img in xb:
            maps = []
            conv_cache = []

            for k in kernels:
                c = convolve2d(img, k, mode="valid")
                a = relu(c)
                p = max_pool(a)
                maps.append(p)
                conv_cache.append((c, a, p))

            feat = np.array(maps).flatten()
            feats.append(feat)
            cache.append(conv_cache)

        feats = np.array(feats)

        h = relu(feats @ W1 + b1)

        logits = h @ W2 + b2
        probs = np.array([softmax(z) for z in logits])

        # LOSS
        y_onehot = np.eye(NUM_CLASS)[yb]
        loss = -np.sum(y_onehot * np.log(probs + 1e-8)) / len(xb)
        total_loss += loss

        pred = np.argmax(probs, axis=1)
        correct += np.sum(pred == yb)

        # -------- BACKPROP ----------
        dlogits = (probs - y_onehot) / len(xb)

        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)

        dh = dlogits @ W2.T * relu_deriv(h)
        dW1 = feats.T @ dh
        db1 = dh.sum(axis=0)

        # propagate to feature maps (before dense)
        dfeats = dh @ W1.T  # shape (B, feat_size)

        # compute gradients for kernels via conv-backprop through pooling + relu
        dKernels = np.zeros_like(kernels)

        for bi in range(len(xb)):
            img = xb[bi]
            conv_caches = cache[bi]  # list of (c, a, p) per kernel
            df = dfeats[bi].reshape(NUM_KERNEL, pool_h, pool_w)

            for ki in range(NUM_KERNEL):
                c, a, p = conv_caches[ki]

                # backprop through max-pool: distribute grad to max locations
                da = np.zeros_like(a)
                for ii in range(pool_h):
                    for jj in range(pool_w):
                        patch = a[ii*POOL_SIZE:(ii+1)*POOL_SIZE, jj*POOL_SIZE:(jj+1)*POOL_SIZE]
                        if patch.size == 0:
                            continue
                        max_val = np.max(patch)
                        mask = (patch == max_val)
                        # if multiple maxima, distribute equally
                        if np.sum(mask) > 0:
                            da_patch = (df[ki, ii, jj] / np.sum(mask)) * mask
                        else:
                            da_patch = 0
                        da[ii*POOL_SIZE:(ii+1)*POOL_SIZE, jj*POOL_SIZE:(jj+1)*POOL_SIZE] += da_patch

                # through ReLU on conv output
                dc = da * relu_deriv(c)

                # gradient w.r.t kernel is cross-correlation of input and dc
                dK = correlate2d(img, dc, mode="valid")
                dKernels[ki] += dK

        # average kernel gradients over batch
        dKernels /= len(xb)

        # Update parameters
        W2 -= LR * dW2
        b2 -= LR * db2
        W1 -= LR * dW1
        b1 -= LR * db1

        kernels -= LR_K * dKernels

    # VALIDASI pada data validasi setiap akhir epoch
    val_pred = []
    for img in X_val:
        maps = []
        for k in kernels:
            c = relu(convolve2d(img, k, mode="valid"))
            maps.append(max_pool(c))

        feat = np.array(maps).flatten()
        h = relu(feat @ W1 + b1)
        p = softmax(h @ W2 + b2)

        val_pred.append(np.argmax(p))

    val_acc = np.mean(val_pred == y_val)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={total_loss:.4f} | "
          f"Acc train={correct/len(X_train):.4f} | "
          f"Acc val={val_acc:.4f}")

# ===================== SIMPAN MODEL =====================
# Simpan bobot dan konfigurasi model ke file .npz
np.savez(
    os.path.join(BASE_DIR, "Trained.npz"),
    kernels=kernels,
    W1=W1, b1=b1,
    W2=W2, b2=b2,
    IMG_W=IMG_W, IMG_H=IMG_H,
    class_names=np.array(class_names)
)

print("[OK] MODEL TERSIMPAN → cnn_manual_224x224aseliii.npz")
