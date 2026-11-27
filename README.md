# Fake News Detection

## Overview

A Python-based machine learning project exploring the classification of news articles as real or fake.

The programme uses the WELFake dataset to train a text classification model, applying TF-IDF vectorisation to convert news articles into numerical representations before using a Passive Aggressive Classifier to distinguish between real and fake information.

## Key Features

- **Data processing:** Loads and prepares the WELFake dataset, handling missing article text and converting numerical labels into categorical classes.
- **Text vectorisation:** Uses Term Frequency-Inverse Document Frequency (TF-IDF) to transform news articles into numerical feature vectors.
- **Machine learning:** Trains a Passive Aggressive Classifier to identify patterns associated with real and fake news.
- **Model evaluation:** Splits the dataset into training and testing sets, measuring performance using accuracy and a confusion matrix.
- **New article classification:** Applies the trained model to a separate collection of news articles and outputs predicted labels.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## How It Works

The programme follows a text classification pipeline:

1. **Dataset preparation:** Loads the WELFake dataset and prepares the article text and corresponding labels.
2. **Train-test split:** Divides the dataset into 80% training data and 20% testing data.
3. **Feature extraction:** Applies TF-IDF vectorisation, using English stop-word removal and a maximum document frequency threshold of 0.7.
4. **Model training:** Trains a Passive Aggressive Classifier on the resulting feature vectors.
5. **Evaluation:** Predicts labels for the test dataset and calculates classification accuracy, alongside a confusion matrix.
6. **New predictions:** Processes a separate CSV file of news articles using the trained vectoriser and classifier, producing a REAL or FAKE prediction for each article.

## Repository Contents

| File | Description |
|---|---|
| `detection.py` | Main classification and evaluation script |
| `WELFake_Dataset.csv.zip` | Dataset used to train and test the model |
| `new_articles.csv` | Additional news articles used to test predictions |
| `new_articles1.csv` | Additional article data |

## Running the Project

Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

Run the programme:

```bash
python detection.py
```

**Note:** The script currently references the dataset and test CSV files through a `fakeNews` subdirectory. Ensure the files are placed in the expected directory or update the file paths in `detection.py` before running.

## Limitations

The model identifies patterns in the text that are associated with the dataset's labels. Its predictions should not be treated as definitive verification of whether a news article is factually accurate.

## Data

The project uses the WELFake dataset for training and evaluation.
