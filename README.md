# Jersey Number Recognition Project
**Course:** COSC 419B/519B  
**Group:** 6  

## Team Members
- Fahd Seddik  
- Arjun Sampat  
- Patrick Agnimel  
- Zaki Pugh-Fradot  
- Amani Lugalla  

## Overview
This repository is a fork of  
[A General Framework for Jersey Number Recognition](https://github.com/mkoshkina/jersey-number-pipeline).

See **[README_original.md](README_original.md)** for the original documentation and architecture details.

## Setup

1. **Environment & Models:**  
   Use the original setup script to install dependencies (ViTPose, PARSeq, SAM) and download pre-trained weights.

   ```bash
   python setup.py
   ```

2. **Dataset:**  
   Use custom script to download the SoccerNet 2023 Challenge data to the `data/` folder.

   ```bash
   python scripts/download_data.py
   ```

3. **Run Evaluation:**  
   Use evatution script

   ```bash
   python evaluate.py --pred my_predictions.json --gt ./data/SoccerNet/test/test_gt.json
   ```
