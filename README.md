# 🔍 MNIST Object Detection with Bounding Box Regression

A multi-task deep learning project that simultaneously performs **digit classification** and **bounding box regression** on MNIST digits placed on larger 75×75 canvas images. Built with TensorFlow and Keras.

---

## 📌 Overview

This project trains a Convolutional Neural Network (CNN) to solve two tasks at once:

1. **Classification** — Identify which digit (0–9) is present in the image.
2. **Bounding Box Regression** — Predict the exact location of the digit within the padded image using normalized `[xmin, ymin, xmax, ymax]` coordinates.

The model is evaluated using standard classification accuracy and **Intersection over Union (IoU)** for bounding box quality.

---

## 🗂️ Project Structure

```
Object_detection.py         # Main script: data loading, model definition, training, evaluation
```

---

## 🧠 Model Architecture

```
Input (75×75×1)
    │
    ▼
Feature Extractor (CNN)
    ├── Conv2D(16, 3×3, ReLU) → AveragePooling2D(2×2)
    ├── Conv2D(32, 3×3, ReLU) → AveragePooling2D(2×2)
    └── Conv2D(64, 3×3, ReLU) → AveragePooling2D(2×2)
    │
    ▼
Dense Layers
    └── Flatten → Dense(128, ReLU)
    │
    ├──▶ Classification Head → Dense(10, Softmax)   [output: class probabilities]
    └──▶ Bounding Box Head  → Dense(4, Linear)      [output: xmin, ymin, xmax, ymax]
```

The model uses a **shared feature backbone** with two separate output heads — one for classification, one for localization.

---

## 📦 Requirements

### Python Version
- Python 3.8+

### Dependencies

Install all required packages with:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib Pillow
```

| Package              | Purpose                                  |
|----------------------|------------------------------------------|
| `tensorflow`         | Model building, training, and inference  |
| `tensorflow-datasets`| Loading the MNIST dataset                |
| `numpy`              | Array manipulation and IoU computation   |
| `matplotlib`         | Visualization of training metrics        |
| `Pillow`             | Drawing bounding boxes on images         |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mnist-object-detection.git
cd Object-Detection-Model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Create a `requirements.txt` with:
> ```
> tensorflow
> tensorflow-datasets
> numpy
> matplotlib
> Pillow
> ```

### 3. Run the Script

```bash
python Object_detection.py
```

The script will:
- Download MNIST automatically via `tensorflow-datasets`
- Preprocess and augment images to 75×75 by randomly padding
- Train the multi-output CNN for 20 epochs
- Display training samples with bounding boxes
- Print validation accuracy
- Plot training metrics
- Visualize predictions with IoU scores

---

## 📊 Dataset

- **Source**: [MNIST](http://yann.lecun.com/exdb/mnist/) via `tensorflow-datasets`
- **Preprocessing**:
  - Original 28×28 digit images are padded to 75×75 with a **random offset** (simulating object localization)
  - Normalized pixel values to `[0, 1]`
  - Bounding box coordinates are **normalized** to `[0, 1]` relative to image dimensions
  - Labels are **one-hot encoded** for 10 classes

---

## ⚙️ Configuration

Key hyperparameters defined at the top of the script or inline:

| Parameter          | Value         | Description                              |
|--------------------|---------------|------------------------------------------|
| `BATCH_SIZE`       | `64`          | Per-replica batch size                   |
| `EPOCHS`           | `20`          | Number of training epochs                |
| `im_width`         | `75`          | Output image width                       |
| `im_height`        | `75`          | Output image height                      |
| `iou_threshold`    | `0.6`         | Threshold for acceptable bounding boxes  |
| Optimizer          | `Adam`        | Default learning rate                    |
| Classification loss| `categorical_crossentropy` | Multi-class classification |
| BBox loss          | `MSE`         | Bounding box regression                  |

---

## 📉 Training Metrics

After training, the following plots are generated:

- **Bounding Box MSE** — Mean squared error of predicted vs. ground truth box coordinates
- **Classification Accuracy** — Per-epoch train and validation accuracy
- **Classification Loss** — Categorical cross-entropy over epochs

---

## 📐 Intersection over Union (IoU)

IoU is computed after training to evaluate bounding box quality:

```
IoU = (Intersection Area) / (Union Area)
```

- A **smoothing factor** (`1e-10`) is added to prevent division by zero.
- An `iou_threshold` of `0.6` is used to flag poor predictions.
- IoU scores are displayed below each image in the visualization grid.

---

## 🖼️ Visualization

Two visualization stages are included:

1. **Before Training** — Displays raw training/validation digits with ground truth bounding boxes.
2. **After Prediction** — Shows 10 random validation samples with:
   - 🔴 Red box: Predicted bounding box
   - 🟢 Green box: Ground truth bounding box
   - IoU score below each image
   - Red label if the classification is incorrect

---

## 🐛 Known Issues / Notes

- **`create_digits_from_local_fonts`**: There is a bug in the reshape line:
  ```python
  # Buggy line:
  font_digits = np.reshape(np.stack(np.split(...), axis=0)[n, 75*75])
  # Should likely be:
  font_digits = np.reshape(np.stack(np.split(...), axis=0), [n, 75*75])
  ```
  This function is defined but not called in the main pipeline, so it does not affect training.

- **`plot_metrics`** uses a global `history` variable — ensure training completes before calling it.

- **`try_gcs=True`** in `tfds.load` attempts to load from Google Cloud Storage (useful in Colab/GCP). On a local machine, it gracefully falls back to a local download.

---

## ☁️ Running on Google Colab

This script is fully compatible with Google Colab. Simply upload `Object_detection.py` and run:

```python
!python Object_detection.py
```

Or copy the code into a notebook cell. Colab provides free GPU acceleration which significantly speeds up training.

---

## 📈 Expected Results

After 20 epochs, you can typically expect:

| Metric                    | Approximate Value |
|---------------------------|-------------------|
| Validation Accuracy       | ~97–99%           |
| Bounding Box MSE          | ~0.001–0.005      |
| Average IoU               | ~0.85–0.95        |

Results may vary slightly due to the random placement of digits in each epoch.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun et al.
- [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/) — Model building framework
- [TensorFlow Datasets](https://www.tensorflow.org/datasets) — Easy dataset loading
