# Question Difficulty Detection

Machine Learning application that predicts the difficulty level of a question from its text, using NLP (TF-IDF) and a trained classification model.

## Features

- Text vectorization with **TF-IDF**
- Feature scaling and label encoding of the difficulty classes
- Pre-trained model loaded from `model.pkl`
- Prediction of the difficulty level of a new question through `app.py`

## Project structure

```
.
├── app.py                  # Application: loads the model and predicts the difficulty
├── model.pkl               # Trained Machine Learning model
├── tfidf.pkl               # TF-IDF vectorizer
├── scaler.pkl              # Feature scaler
├── label_encoder.pkl       # Label encoder for the difficulty classes
├── questions_dataset.csv   # Dataset of questions used for training
├── master.txt              # (à préciser)
└── requirements.txt        # Python dependencies
```

## Tech stack

- Python
- Machine Learning / NLP
- TF-IDF
- scikit-learn (à confirmer selon ton `requirements.txt`)

## Installation

```bash
git clone https://github.com/garoka/<nom-du-depot>.git
cd <nom-du-depot>
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

> Si ton application utilise Streamlit ou Flask, remplace la commande par `streamlit run app.py` ou `flask run`.

## Dataset

The model is trained on `questions_dataset.csv`, a dataset of questions labeled by difficulty level.

## Author

**Oussama Saidani** - Full-Stack & DevOps Engineering student at ESPRIT
GitHub: [garoka](https://github.com/garoka)
