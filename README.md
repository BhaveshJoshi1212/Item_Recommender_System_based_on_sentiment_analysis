# Item_Recommender_System_based_on_sentiment_analysis


## Project Overview
Developed an advanced recommendation system integrating **collaborative filtering** and **sentiment analysis** to deliver personalized product suggestions based on user reviews. This system combines multiple machine learning and NLP techniques to enhance recommendation accuracy and relevance.


## Major Techniques Used

- **Natural Language Processing (NLP):**
  - **Text Preprocessing:** 
    - Tokenization: Breaking text into individual words or tokens.
    - Stop Word Removal: Eliminating common words that do not contribute much to the meaning of the text.
    - Lemmatization/Stemming: Reducing words to their base or root form.
  - **TF-IDF Vectorization:**
    - Term Frequency-Inverse Document Frequency: Converting text data into numerical features based on word importance. Key parameters:
      - `max_features=650`: Limits the number of features to the top 650 terms.
      - `ngram_range=(1,2)`: Considers both single words and bi-grams.
      - `binary=True`: Represents word presence as binary values.

- **Machine Learning Algorithms:**
  - **Logistic Regression:** Used for binary classification of sentiment (positive/negative).
  - **Naive Bayes:** Applied for sentiment classification, particularly effective for text data.
  - **Decision Tree:** Utilized for its interpretability and ability to handle non-linear data.
  - **Random Forest:** An ensemble method improving classification performance through multiple decision trees.
  - **XGBoost:** A powerful boosting algorithm for improved prediction accuracy and efficiency.

- **Class Imbalance Handling:**
  - **SMOTE (Synthetic Minority Over-sampling Technique):**
    - Generates synthetic samples for the minority class to balance the dataset and improve model performance.

- **Collaborative Filtering:**
  - **User-User Collaborative Filtering:**
    - Computes user similarity using **cosine similarity** to recommend products based on similar users’ preferences.
  - **Item-Item Collaborative Filtering:**
    - Computes item similarity to recommend products based on the similarity between items.

- **Evaluation Metrics:**
  - **Accuracy:** Measures the proportion of correct predictions.
  - **Precision:** Evaluates the accuracy of positive predictions.
  - **Recall:** Measures the ability to find all relevant positive instances.
  - **F1 Score:** Harmonic mean of precision and recall.
  - **ROC-AUC:** Evaluates the model’s performance in distinguishing between classes.



## Brief Processes

1. **Data Collection and Preprocessing:**
   - **Data Gathering:** Acquired user reviews, product ratings, and user information.
   - **Text Preprocessing:** Utilized **Natural Language Processing (NLP)** techniques for text cleaning, including tokenization, stop word removal, and part-of-speech tagging to prepare the data for analysis.
   - **TF-IDF Vectorization:** Employed **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorizer to transform text data into numerical features, capturing the significance of words within documents.
2. **Feature Extraction:**
   - **Class Imbalance Handling:** Addressed class imbalance in sentiment labels using **SMOTE (Synthetic Minority Over-sampling Technique)** to create a balanced dataset for model training.
   - **Feature Engineering:** Applied TF-IDF to convert text reviews into feature vectors suitable for machine learning models.

3. **Model Building:**
   - **Classification Algorithms:** Implemented and compared various **machine learning algorithms** including Logistic Regression, Naive Bayes, Decision Tree, Random Forest, and XGBoost for sentiment classification.
   - **Evaluation Metrics:** Evaluated models using **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **ROC-AUC**. Used **confusion matrix** and **ROC curve** for performance assessment.
   - **Hyperparameter Tuning:** Optimized Random Forest parameters using **Grid Search** to enhance model performance.

4. **Collaborative Filtering Approaches:**
   - **User-User Collaborative Filtering:** Developed a User-User recommendation model based on user similarity using **cosine similarity** to compute user-user distances and predict ratings.
   - **Item-Item Collaborative Filtering:** Built an Item-Item recommendation model focusing on item similarity, calculating item-item distances to recommend products not yet rated by users.

5. **Sentiment-Based Recommendation Integration:**
   - **Sentiment Prediction:** Integrated **sentiment analysis** using an XGBoost model to classify review sentiments (positive or negative).
   - **Recommendation Enhancement:** Refined product recommendations by combining collaborative filtering results with sentiment analysis, prioritizing products with higher positive sentiment scores.

6. **Evaluation and Comparison:**
   - **Model Comparison:** Compared performance of different recommendation models using **RMSE (Root Mean Squared Error)** to identify the most effective model.
   - **Recommendation System Integration:** Enhanced recommendations by integrating sentiment analysis results with collaborative filtering to improve relevance and accuracy.



## Key Achievements
- Successfully combined **collaborative filtering** and **sentiment analysis** to improve recommendation quality.
- Demonstrated practical application of **NLP** techniques and **machine learning** algorithms in a real-world recommendation system.
- Enhanced recommendation relevance through integration of sentiment analysis, providing more personalized product suggestions.

## Tags
`#MachineLearning` `#NLP` `#RecommendationSystems` `#CollaborativeFiltering` `#SentimentAnalysis` `#TF-IDF` `#SMOTE` `#XGBoost`
