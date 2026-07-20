# Flower Classification using VGG16

## Overview

This project classifies flower images using a pretrained VGG16 Convolutional Neural Network with Transfer Learning.

## Features

- Image Classification
- Flask Web Application
- REST API
- Azure App Service Deployment
- GitHub Actions CI/CD

## Classes

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

## Run Locally

```bash
pip install -r requirements.txt
python main.py
```

Open:

```
http://127.0.0.1:5000
```

API:

```
POST /api/predict
```