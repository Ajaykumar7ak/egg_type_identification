# Egg Type Classifier — FAST Starter Kit

This project fine-tunes a pretrained CNN to classify egg images into 5 types:
**chicken, duck, turkey, emu, ostrich**. It includes:
- Dataset splitter
- Training (transfer learning + fine-tuning)
- Evaluation (report + confusion matrix)
- Flask web app for image upload & prediction

## 1) Put your images
Place your images inside `data/raw` using **one folder per class**:
```
data/raw/
  chicken/
  duck/
  turkey/
  emu/
  ostrich/
```
> Folder name = label. Add `.jpg/.jpeg/.png` images.

## 2) (Optional) Split into train/val/test automatically
```
python split_dataset.py --raw_dir data/raw --out_dir data --train 0.7 --val 0.2 --test 0.1
```

## 3) Install & Train
Create a virtual environment (Windows PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
macOS/Linux:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Verify GPU (optional):**
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Train:**
```
python train.py --data_dir data --epochs1 10 --epochs2 10 --batch 16 --img 224
```
Artifacts saved in `models/`:
- `egg_model.keras` (final model)
- `label_map.json` (index → label)
- `classification_report.txt`
- `confusion_matrix.png`

## 4) Run the Web App
```
python app.py
```
Open http://127.0.0.1:5000, upload an egg image, see prediction.

---

### Tips for Your Small Dataset (~316 images)
- Keep classes balanced if possible.
- Data augmentation is already enabled.
- You can increase `--epochs2` for better accuracy after first stage.
- If VRAM is tight, try `--batch 8` or `--img 192`.

Good luck & have fun!
