# Extending HAMMER with Segmentation-Guided Scoring (SGS)

This repository is for project demonstration purposes.  
It contains scripts and instructions to reproduce the lightweight **Segmentation-Guided Scoring (SGS)** pipeline for detecting global foreground–background (FG–BG) inconsistencies in multimodal media.

---

## Repository Structure

```
hammer-sgs/
├─ scripts/                  # Core Python scripts for SGS
│  ├─ semantic_fg_bg_blip.py   # BLIP + MiniLM semantic similarity
│  └─ yolo_seg_union_people.py     # YOLOv8-seg union masks for people
├─ data/                     
├─ results/                  # Example output CSVs (gitignored by default)
├─ requirements.txt          # Python dependencies
├─ .gitignore                # Ignore caches, large files, outputs
└─ README.md                 # This file
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Gaganx0/hammer-sgs.git
cd hammer-sgs
```

### 2. Create a Python environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 3. Prepare the dataset
We rely on the **DGM4plus** dataset:  
https://github.com/Gaganx0/DGM4plus  

Download it and place it locally. Expected layout:

```
/path/to/DGM4plus/
  origin/...          
  manipulation/...    
  metadata.json       
```

### 4. Generate FG–BG crops with YOLOv8-seg
```bash
python scripts/yolo_seg_union_people.py   --data /path/to/DGM4plus/metadata.json   --img_root /path/to/DGM4plus   --out_dir crops   --model yolov8s-seg.pt   --imgsz 896
```
This produces:
- `crops/fg/<id>.png` → Foreground crop (all people merged)  
- `crops/bg/<id>.png` → Background crop (inverse mask)

### 5. Run the SGS semantic checker
```bash
python scripts/semantic_fg_bg_blip.py   --fg_dir crops/fg   --bg_dir crops/bg   --out_csv results/semantic_blip_cpu.csv   --threshold 0.55
```
The output CSV will contain:
```
id, fg_path, bg_path, fg_text, bg_text, sts01, label
```
- `sts01`: similarity score (0–1)
- `label`: "Match" or "Mismatch"

---

## Example Output

```
id,fg_path,bg_path,fg_text,bg_text,sts01,label
001,...,...,"a man in a suit","a snowy mountain",0.21,Mismatch
002,...,...,"a woman teaching","a classroom",0.78,Match
```

---

## Requirements

See `requirements.txt`. Key libraries:
- `torch`, `torchvision`
- `transformers`, `sentence-transformers`
- `ultralytics` (YOLOv8)
- `opencv-python`, `Pillow`, `tqdm`, `pandas`

---

## Notes
- This is a **lightweight demo repo**; no paper source or heavy configs are included.
- Threshold `--threshold` can be tuned per dataset for sensitivity vs. specificity.

---

## License
MIT License © 2025

---

## Citation
If you use this code or ideas in research, please cite the __ paper (to appear).
