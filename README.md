# News-Category-Classification-using-Transformers

# 1. Dataset Link

Dataset Name: News Category Dataset (HuffPost)

Source: Kaggle

Link: https://www.kaggle.com/datasets/rmisra/news-category-dataset

Dataset Description:

Around 210,000 news headlines and short descriptions

Each sample is labeled with one of 41 news categories

Data is provided in JSON format

# 2. Model Used

Model: *distilbert-base-uncased*

Library: Hugging Face Transformers

Task: Multi-class text classification (41 categories)

Why DistilBERT was chosen:

Faster and lighter than BERT

Suitable for short texts like news headlines

Good trade-off between performance and efficiency

# 3. How to Train the Model

Model training was done in Google Colab using GPU.

Training Steps:

Load the dataset JSON file

Combine headline and short_description into a single text field

Encode category labels using LabelEncoder

Split the data into train and validation sets (80/20, stratified)

Tokenize text using the DistilBERT tokenizer (max_length = 128)

Fine-tune the pretrained model using Hugging Face Trainer

Evaluate the model on the validation set

Save the trained model, tokenizer, and label mapping

## Training Summary:

Training samples: 94,829

Validation samples: 23,708

Epochs: 1

## Evaluation Metrics:

Validation Accuracy: 69.28%

Validation F1-score (weighted): 68.50%

A confusion matrix was also inspected to understand category-level errors.

# 4. How to Run the Backend API

The backend is built using FastAPI and loads the trained model for inference.

Install Dependencies
pip install -r requirements.txt

Run the API Server

From the project root directory:

uvicorn backend.app:app --reload

Access API Documentation

Open in browser:

http://127.0.0.1:8000/docs

Inference Endpoint

POST /predict

Request Body:

{
  "text": "Apple announces new AI-powered MacBooks"
}


Response Body:

{
  "predicted_category": "TECH",
  "confidence": 0.97
}

# 5. How the Database Is Structured

The backend uses SQLite to store inference requests.

Database File
predictions.db

| Column             | Description                 |
|--------------------|-----------------------------|
| id                 | Auto-increment primary key  |
| input_text         | Text received by the API    |
| predicted_category | Model prediction            |
| confidence         | Prediction confidence score |
| created_at         | Timestamp of the request    |


The database is used only for logging inference requests and does not store any training data.

Project Structure
```
news-category-classifier/
├── backend/
│   ├── app.py
│   ├── database.py
│   └── models.py
├── model/
│   ├── classifier/
│   ├── tokenizer/
│   └── label_map.json
├── training/
│   └── train_and_evaluate.py 
├── requirements.txt
└── README.md
```


Final Note

This project demonstrates an end-to-end machine learning workflow, from transformer fine-tuning to deploying a production-style inference API with database persistence.
