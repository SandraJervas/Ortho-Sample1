# Orthodontic Image Classification API

## Project Overview
This project trains a ResNet-18 model to classify orthodontic images into 8 classes (frontal at rest, frontal smile, intraoral views, profile at rest, etc.). The model is exposed as a REST API using FastAPI.

## Features
- Training script for model training and validation
- Prediction API endpoint accepting image URLs
- Duplicate detection with perceptual hashing
- Confidence thresholding and logging of uncertain predictions

## Usage

### Training
- Prepare your dataset folders: train, val with subfolders for each class
- Run `python train.py` to train the model
- Model weights saved to the path specified in `params.yaml`

### Running API Server
- Run `uvicorn src.api.app:app --reload` to start the FastAPI server
- Use `/predict` POST endpoint with JSON `{"url": "<image_url>"}` to get predictions

## Class Names
- frontal_at_rest
- frontal_smile
- intraoral_front
- intraoral_left
- intraoral_right
- lower_jaw_view
- profile_at_rest
- upper_jaw_view

## Logs and Outputs
- Duplicate image hashes stored in `hash_store.json`
- Low confidence predictions logged to `uncertain_log.csv`
- Latest downloaded image saved as `last_downloaded.jpg`

## Requirements
- Python 3.8+
- PyTorch
- FastAPI
- torchvision
- imagehash
- other dependencies listed in `requirements.txt`

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/SandraJervas/Ortho-Sample1.git
cd Ortho-Sample1
