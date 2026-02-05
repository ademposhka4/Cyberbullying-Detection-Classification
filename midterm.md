---
layout: default
title: "Midterm"
permalink: /midterm/
---

<nav class="tabs">
  <a href="{{ '/project-proposal/' | relative_url }}"
     class="tab{% if page.url == '/project-proposal/' or page.url == '/' %} active{% endif %}">
    Project Proposal
  </a>
  <a href="{{ '/midterm/' | relative_url }}"
     class="tab{% if page.url == '/midterm/' %} active{% endif %}">
    Midterm
  </a>
  <a href="{{ '/final/' | relative_url }}"
     class="tab{% if page.url == '/final/' %} active{% endif %}">
    Final Report
  </a>
</nav>


# *Cyberbullying Detection*
# Introduction/Background
Cyberbullying is a major social issue in our technology-dependent world that induces severe psychological consequences for its victims. The term is defined as the use of digital devices and the internet to intentionally harm, harass, humiliate, and outright bully someone, often through social media, text messages, or emails [1]. Its nature on social media platforms warrants automated detection methods to ensure safer online environments for everyone. This project will focus on developing a machine learning model for multi-class cyberbullying classification based on documented social media comments and posts.

## Literature Review
As of now, the field of automated cyberbullying detection is heavily reliant on Natural Language Processing (NLP) and Machine Learning techniques. Earlier research efforts have experimented with traditional machine learning models, such as Support Vector Machines and Naive Bayes, which utilize lexicon-based and n-gram features [2]. New approaches should be taken, however, since the increasingly complex, noisy, and evolving slang characteristic of online harassment challenged these previous methods. More recent literature has pointed to the improved performance of Deep Learning architectures in cyberbullying detection, like Recurrent Neural Networks and Long Short-Term Memory Networks, which automatically pinpoint unique features from complex text sequences [3]. Additionally, recent studies have advanced beyond the realm of binary (bullying or non-bullying) detection to a more complex multi-class classification specific target types of the documented cyberbullying cases. Frameworks that incorporate pre-trained models like Bidirectional Encoder Representations from Transformers have shown improved results for this increasingly complex task [4]. These newer models capture contextual embeddings, which significantly improve the identification of subtle and context-dependent forms of cyberbullying that previous models could not detect.

## Dataset Description
This project will utilize a Cyberbullying Classification dataset from Kaggle, a collection of thousands of social media comments and posts that will serve as the core textual data for classification. The dataset includes a collection of raw data split into two categories: tweet text and cyberbullying type. This will require NLP techniques to preprocess and vectorize the data for multi-class classification. The dataset can be found here: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

# Problem Definition:

**Problem**  
Social media users are experiencing unprecedented levels of cyberbullying, which has significantly harmed the mental health of millions in recent years. These impacts can lead to serious consequences for both young people and adults alike.  

**Motivation**  
By leveraging machine learning, we can automatically classify instances of cyberbullying, specifically the use of inappropriate language and derogatory terms directed at other users. This would allow us to proactively block offenders, thereby reducing the harmful effects of online harassment. In turn, this approach helps create a safer, more respectful, and cleaner social media environment for all users.

# Methods

**Data Preprocessing**  
- Remove symbols like “@” and “#” to reduce noise while maintaining relevant information  
- Remove emojis and lowercase all words because they are strong sentiment cues and neutralizing them exposes them to n-gram features  
- Character n-grams to catch underscores or other obstructions  

**ML Algorithms**  
- **Linear SVM** (`sklearn.svm.LinearSVC`) - Effective for high dimensional sparse text; very fast and strong.  
- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`) - Regression handles the dataset format well with deciding features that shift the weights to the proper classification. They also output meaningful probabilities.  
- **Complement Naive Bayes** (`sklearn.naive_bayes.ComplementNB`) - Assumes conditional independence of features and works well under skewed term distributions which is good when dealing with offensive language that can dominate social media  

# Results and Discussion

Ultimately, the goal of our model is to effectively flag potentially harmful tweets, as well as be able to classify them by their type of cyberbullying. In order to evaluate how effective our model is, we used the following quantitative metrics. 

**Metrics**

1. **Accuracy Score (`accuracy_score`)** – Computes the accuracy of correctly classified tweets. Accuracy will give us a general sense of our model’s performance.  

2. **F1 Score (`f1_score`)** – Balances precision and recall across all categories of tweets. This will provide us with a better evaluation across all classes compared to accuracy.  

3. **Confusion Matrix (`confusion_matrix`)** – Evaluates classification accuracy and helps us identify patterns of misclassification. This will allow us to see if categories are confused with each other.  

4. **Log Loss (`log_loss`)** – Penalizes overconfident, incorrect predictions. This allows us to evaluate the classes that may not have the best accuracy.


**Goals**

In terms of the metrics, our goal was to achieve at least a 75% F1 score while having relatively balanced accuracy between all categories of cyberbullying. Using the confusion matrix, we aimed to minimize misclassifications, particularly false negatives as instances of bullying that are undetected can potentially lead to harm. On the other hand, we didn’t want to over flag non-harmful tweets as cyberbullying, as this can lead to censorship issues.

**Logistic Regression Results**

Our baseline logistic regression model performed well, providing us with an accuracy score of 0.83. Additionally, we achieved an F1 score of 83%, which surpassed our goal of 75%. Based on these results and our data set, our model performed very well in terms of correctly predicting outcomes. Additionally, the F1 score suggests that for most classes, the model had high precision and recall. To further verify, we used a confusion matrix. The confusion matrix suggests that we had good results for the age, ethnicity, gender, and religion classes. It performed reasonably for the non-cyberbullying and other cyberbullying classes, but the results of these two classes were not as strong as the other classes. The model seemed to sometimes have confusion between these two classes. To further evaluate the model, we used log loss to make sure that it wasn’t confidently incorrectly predicting non-cyberbullying or other cyberbullying. The log loss score of 0.41 suggests that the model is performing well.

![Logistic Regression Confusion]({{ '/assets/logistic_regression_confusion.png' | relative_url }})

**Linear SVM Results**

Our linear SVM model performed slightly better on average, with an accuracy and F1 score of 85%, again passing our target threshold of 75%. This model outperformed our logistic regression model across all metrics, with modest gains across most classes. We can infer from this that the linear SVM model provided us with more effective decision boundaries and is potentially a better model compared to logistic regression. The confusion matrix for this model also shows promising results for the age, ethnicity, gender, and religion classes, as well as for the main classification between cyberbullying and non-cyberbullying tweets. Although the model’s determination of which tweets are cyberbullying outperformed our logistic regression model, non-cyberbullying remains the least accurate class.

![Linear SVM Confusion]({{ '/assets/LinearSVM_Confusion.png' | relative_url }})

**Next Steps**

We will iterate on our models again to see if we can achieve better results. Specifically, we will try to improve the accuracy and F1 scores for the non cyberbullying and other cyberbullying classes. To do so we will explore improving our feature representations. We may also explore more complex models that may outperform the current models we have. 

# References
[1] U.S. Department of Health and Human Services, “What Is Cyberbullying,” stopbullying.gov, Nov. 05, 2021. https://www.stopbullying.gov/cyberbullying/what-is-it  

[2] Aljwharah Alabdulwahab, Mohd Anul Haq, and M. Alshehri, “Cyberbullying Detection using Machine Learning and Deep Learning,” International journal of advanced computer science and applications/International journal of advanced computer science & applications, vol. 14, no. 10, Jan. 2023, doi: https://doi.org/10.14569/ijacsa.2023.0141045.  

[3] Mst Shapna Akter, Hossain Shahriar, N. Ahmed, and A. Cuzzocrea, “Deep Learning Approach for Classifying the Aggressive Comments on Social Media: Machine Translated Data Vs Real Life Data,” 2022 IEEE International Conference on Big Data (Big Data), Dec. 2022, doi: https://doi.org/10.1109/bigdata55660.2022.10020249.  

[4] P. Aggarwal and R. Mahajan, “Shielding Social Media: BERT and SVM Unite for Cyberbullying Detection and Classification,” Journal of Information Systems and Informatics, vol. 6, no. 2, pp. 607–623, Jun. 2024, doi: https://doi.org/10.51519/journalisi.v6i2.692.

# Collaboration
## Current Responsibilities
Please see our [Gantt Chart](CS 4641 Gantt Chart.xlsx)

# Contributions

| **Name**          | **Proposal Contributions**                         |
|-------------------|-----------------------------------------------------|
| Lucas Paschke     | Model Selection, Github Page Updates, Github Repo Mangement|
| Gabe Ashkenazi    | Data Pre-Processing, Model 1 Data Sourcing, Cleaning, and implementation|
| Matthew Bressler  | Evaluate results and metrics, next steps|
| Ivan Yakubovich   | Data Pre-Processing, Model 2 Data Sourcing, Cleaning, and implementation|
| Adem Poshka       | Evaluate results and metrics, next steps|
