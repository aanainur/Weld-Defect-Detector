import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import cv2
import os
from scipy.signal import convolve2d
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ===================== KONFIGURASI GUI =====================
# Path model bisa disesuaikan jika model berada di folder lain
BASE_DIR = r"c:\Users\..."
MODEL_PATH = os.path.join(BASE_DIR, "Trained.npz")

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

# ===================== MUAT MODEL =====================
# Fungsi ini memuat parameter model CNN yang telah dilatih dari file .npz
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
    
    return model

# ===================== PREPROCESSING =====================
# Preprocessing citra sebelum dikirim ke model CNN
def preprocess_image_with_steps(img_path, img_w, img_h):
    """Load and preprocess image, return original and preprocessed"""
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img_original is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = img_original.copy()
    
    # Resize citra agar sesuai ukuran input model
    img = cv2.resize(img, (img_w, img_h))
    img_resized = img.copy()
    
    # Denoising dengan Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_blurred = img.copy()
    
    # Normalisasi ke rentang 0-1
    img = img / 255.0
    img_normalized = img.copy()
    
    return {
        'original': img_original,
        'resized': img_resized,
        'blurred': img_blurred,
        'normalized': img_normalized
    }

# ===================== INFERENCE =====================
# Fungsi untuk melakukan prediksi kelas dari citra input
def predict(model, img_path):
    """Predict class for given image"""
    
    # Preprocessing citra input
    imgs = preprocess_image_with_steps(img_path, model['IMG_W'], model['IMG_H'])
    img = imgs['normalized']
    
    kernels = model['kernels']
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    
    # Forward pass manual ke CNN
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
        'all_classes': model['class_names'],
        'images': imgs
    }

# ===================== GUI =====================
class CNNPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Image Classifier - Welding Defects")
        self.root.geometry("1200x800")
        
        self.model = None
        self.current_prediction = None
        
        # Load model
        try:
            self.model = load_model(MODEL_PATH)
            self.load_status = f"Model loaded: {self.model['IMG_W']}x{self.model['IMG_H']}"
        except Exception as e:
            self.load_status = f"Error: {str(e)}"
            messagebox.showerror("Model Load Error", self.load_status)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI components"""
        
        # Top Frame - Controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(top_frame, text="Load Image:").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Browse File", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="From Validation Folder", command=self.load_from_folder).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text=self.load_status, foreground="green").pack(side=tk.LEFT, padx=20)
        
        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Image display (Before/After)
        left_frame = ttk.LabelFrame(content_frame, text="Image Preprocessing", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Before preprocessing
        before_label = ttk.Label(left_frame, text="Original Image:")
        before_label.pack()
        self.canvas_before = tk.Canvas(left_frame, width=250, height=250, bg='gray')
        self.canvas_before.pack(padx=5, pady=5)
        
        # After preprocessing
        after_label = ttk.Label(left_frame, text="After Preprocessing:")
        after_label.pack()
        self.canvas_after = tk.Canvas(left_frame, width=250, height=250, bg='gray')
        self.canvas_after.pack(padx=5, pady=5)
        
        # Right: Prediction results
        right_frame = ttk.LabelFrame(content_frame, text="Prediction Results", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Prediction class + confidence
        prediction_frame = ttk.Frame(right_frame)
        prediction_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(prediction_frame, text="Predicted Class:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.label_class = ttk.Label(prediction_frame, text="---", font=("Arial", 24, "bold"), foreground="blue")
        self.label_class.pack(anchor=tk.W, pady=5)
        
        ttk.Label(prediction_frame, text="Confidence:", font=("Arial", 11)).pack(anchor=tk.W)
        self.label_confidence = ttk.Label(prediction_frame, text="---", font=("Arial", 14))
        self.label_confidence.pack(anchor=tk.W, pady=5)
        
        # Probability bars for all classes
        ttk.Label(right_frame, text="Class Probabilities:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(20, 10))
        
        self.prob_frame = ttk.Frame(right_frame)
        self.prob_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prob_bars = {}
        if self.model:
            for cls_name in self.model['class_names']:
                cls_frame = ttk.Frame(self.prob_frame)
                cls_frame.pack(fill=tk.X, pady=5)
                
                cls_label = ttk.Label(cls_frame, text=cls_name, width=15, anchor=tk.W)
                cls_label.pack(side=tk.LEFT, padx=5)
                
                progress = ttk.Progressbar(cls_frame, mode='determinate', length=300, maximum=100)
                progress.pack(side=tk.LEFT, padx=5)
                
                prob_text = ttk.Label(cls_frame, text="0.0%", width=8, anchor=tk.E)
                prob_text.pack(side=tk.LEFT, padx=5)
                
                self.prob_bars[cls_name] = {
                    'progress': progress,
                    'text': prob_text
                }
    
    def load_image(self):
        """Load single image from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
            initialdir=BASE_DIR
        )
        
        if file_path:
            try:
                self.predict_and_display(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def load_from_folder(self):
        """Load random image from validation folder"""
        val_folder = os.path.join(BASE_DIR, "valid")
        
        # Pilih kelas dan file secara acak dari folder validasi
        classes = sorted(os.listdir(val_folder))
        import random
        cls = random.choice(classes)
        cls_folder = os.path.join(val_folder, cls)
        
        images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not images:
            messagebox.showwarning("Warning", "No images found in validation folder")
            return
        
        img_file = random.choice(images)
        img_path = os.path.join(cls_folder, img_file)
        
        try:
            self.predict_and_display(img_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def predict_and_display(self, img_path):
        """Predict and display results"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        result = predict(self.model, img_path)
        self.current_prediction = result
        
        # Tampilkan citra asli dan setelah preprocessing
        self.display_image_on_canvas(result['images']['original'], self.canvas_before)
        self.display_image_on_canvas((result['images']['normalized'] * 255).astype(np.uint8), self.canvas_after)
        
        # Update teks hasil prediksi pada UI
        self.label_class.config(text=result['class'])
        self.label_confidence.config(text=f"{result['confidence']:.2f}%")
        
        # Update bar probabilitas setiap kelas
        for cls_name, prob in zip(result['all_classes'], result['all_probs']):
            prob_percent = prob * 100
            self.prob_bars[cls_name]['progress'].config(value=prob_percent)
            self.prob_bars[cls_name]['text'].config(text=f"{prob_percent:.1f}%")
    
    def display_image_on_canvas(self, img, canvas):
        """Display numpy image on tkinter canvas"""
        # Convert to uint8 if not already
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Resize to fit canvas
        h, w = img.shape[:2]
        scale = min(250 / h, 250 / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Convert grayscale to RGB for PIL
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_resized
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Create PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display on canvas
        canvas.create_image(125, 125, image=photo)
        canvas.image = photo  # Keep reference to prevent garbage collection

def main():
    root = tk.Tk()
    app = CNNPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
