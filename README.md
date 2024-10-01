# **ADHD Detection Using Textual Data Analysis**

This project aims to detect the likelihood of ADHD (Attention Deficit Hyperactivity Disorder) by analyzing textual data, particularly from user posts on Reddit. The project involves both machine learning (ML) and deep learning (DL) approaches to extract meaningful features from text and predict ADHD likelihood.

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Approach](#machine-learning-approach)
- [Deep Learning Approach](#deep-learning-approach)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## **Project Overview**
The aim of this project is to build a predictive model that detects ADHD based on textual data. Traditional ADHD diagnoses rely on medical tests like MRIs or behavioral assessments, which may not always be accessible or affordable. By analyzing the language patterns of individuals, this model provides a more accessible method for predicting ADHD based on everyday text.

We utilize Reddit posts, where users discuss their daily struggles with ADHD. From these texts, key features are extracted and used to train both machine learning and deep learning models for ADHD prediction.

## **Dataset**
The dataset used for this project was scraped from Reddit, containing posts from users discussing ADHD. The dataset includes two primary columns:
- `title`: The title of the post.
- `selftext`: The body of the post.

### **Preprocessing**
- Text cleaning (removal of stop words, punctuation, etc.)
- Feature extraction (explained below)

## **Feature Extraction**
To make predictions, various features were extracted from the text data:
1. **ADHD-Related Keyword Frequency**: Counts how often ADHD-related terms (e.g., "hyperactivity", "impulsivity") are used.
2. **Sentiment Analysis**: Using TextBlob, we calculate the polarity score (range: -1 to 1) to understand the tone of the text.
3. **Topic Modeling**: Using Latent Dirichlet Allocation (LDA) with TF-IDF, we model hidden topics in the text.
4. **Named Entity Recognition (NER)**: Extracts key entities using spaCy.
5. **Text Complexity**: Measured with the Flesch-Kincaid grade level to assess the complexity of the text.
6. **Contextual Relevance**: Calculated using cosine similarity to ADHD-specific keywords.
7. **Emotional Tone**: Using VADER sentiment analysis to calculate a compound score (range: -1 to 1) reflecting emotional tone.

These features are used as inputs for both machine learning and deep learning models.

## **Machine Learning Approach**
We trained several machine learning models using the extracted features:
- **Model**: Random Forest Classifier
- **Features Used**: All seven features mentioned above.
- **Accuracy**: 99%

We tuned the hyperparameters and validated the model using cross-validation techniques to optimize performance.

## **Deep Learning Approach**
We also developed a deep learning model using neural networks to predict ADHD based on the same features.
- **Model**: Sequential Neural Network (Keras)
- **Architecture**: 
  - 512 units with ReLU activation + Dropout
  - 256 units with ReLU activation + Dropout
  - 128 units with ReLU activation + Dropout
  - 64 units with ReLU activation
  - Final softmax layer for multi-class classification
- **BERT Embeddings**: Integrated BERT embeddings for better text representation.
- **Accuracy**: Achieved competitive accuracy in comparison to the ML model.

## **Results**
- **Machine Learning Model Accuracy**: 99%
- **Deep Learning Model Accuracy**: Comparable to the ML model, leveraging BERT embeddings and more complex architectures for nuanced text understanding.

Both models provided strong results, showing the potential of language-based analysis for predicting ADHD.

## **Installation**
To use this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/adhd-text-analysis.git
   cd adhd-text-analysis
   ```

2. **Install dependencies**:
   Create a virtual environment and install the required packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:
   If using BERT embeddings, download pre-trained models from the Hugging Face model hub.

4. **Run the model**:
   ```bash
   python train_model.py
   ```

## **Usage**
1. **Data Preprocessing**: Clean the dataset and extract the features.
   ```bash
   python preprocess_data.py
   ```

2. **Train Machine Learning Model**: Train the Random Forest Classifier using the extracted features.
   ```bash
   python train_ml_model.py
   ```

3. **Train Deep Learning Model**: Train the neural network for ADHD prediction.
   ```bash
   python train_dl_model.py
   ```

4. **Evaluate Models**: Run evaluations to check model performance on test data.
   ```bash
   python evaluate_model.py
   ```

## **Contributing**
If you want to contribute to the project, feel free to fork the repository and submit pull requests. Any contributions are welcome.


---

Feel free to modify the README as needed based on your project’s specifics!
