---
layout: default
title: "Project Proposal"
permalink: /project-proposal/
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
Ultimately, the goal of our model is to effectively flag potentially harmful tweets, as well as be able to classify them by their type of cyberbullying. In order to evaluate how effective our model is, we must use quantitative metrics.

**Metrics**  
- **Accuracy Score** (`accuracy_score`) - computes the accuracy of correctly classified tweets. Accuracy will give us a general sense of our model’s performance.  
- **F1 Score** (`f1_score`) - balances precision and recall across all categories of tweets. This will provide us with a better evaluation across all classes compared to accuracy.  
- **Confusion Matrix** (`confusion_matrix`) - evaluates classification accuracy and helps us identify patterns of misclassification. This will allow us to see if categories are confused with each other.  

**Goals**  
In terms of the metrics, our goal is to achieve at least a 75% F1 score while having relatively balanced accuracy between all categories of cyberbullying. Using the confusion matrix, we aim to minimize misclassifications, particularly false negatives as instances of bullying that are undetected can potentially lead to harm. On the other hand, we don’t want to over flag non-harmful tweets as cyberbullying, as this can lead to censorship issues.

**Expected Results**  
We expect Linear SVM to perform the best among the algorithms due to its strong ability to comprehend high dimensional text. We also expect Logistic Regression to provide us with a strong foundation, giving us the likelihood that a tweet is in a certain category.

# References
[1] U.S. Department of Health and Human Services, “What Is Cyberbullying,” stopbullying.gov, Nov. 05, 2021. https://www.stopbullying.gov/cyberbullying/what-is-it  

[2] Aljwharah Alabdulwahab, Mohd Anul Haq, and M. Alshehri, “Cyberbullying Detection using Machine Learning and Deep Learning,” International journal of advanced computer science and applications/International journal of advanced computer science & applications, vol. 14, no. 10, Jan. 2023, doi: https://doi.org/10.14569/ijacsa.2023.0141045.  

[3] Mst Shapna Akter, Hossain Shahriar, N. Ahmed, and A. Cuzzocrea, “Deep Learning Approach for Classifying the Aggressive Comments on Social Media: Machine Translated Data Vs Real Life Data,” 2022 IEEE International Conference on Big Data (Big Data), Dec. 2022, doi: https://doi.org/10.1109/bigdata55660.2022.10020249.  

[4] P. Aggarwal and R. Mahajan, “Shielding Social Media: BERT and SVM Unite for Cyberbullying Detection and Classification,” Journal of Information Systems and Informatics, vol. 6, no. 2, pp. 607–623, Jun. 2024, doi: https://doi.org/10.51519/journalisi.v6i2.692.

### Word Count: 774 (Excluding Headers)

# Collaboration
## Planned Responsibilities
Please see our [Gantt Chart](CS 4641 Gantt Chart.xlsx)

# Contributions

| **Name**          | **Proposal Contributions**                         |
|-------------------|-----------------------------------------------------|
| Lucas Paschke     | Introduction/Background, GitHub page, Dataset selection |
| Gabe Ashkenazi    | Methods, Gantt chart, Dataset selection             |
| Matthew Bressler  | Results and Discussion                              |
| Ivan Yakubovich   | Problem Definition                                  |
| Adem Poshka       | Video Presentation                                  |

**We are opting in to the award consideration**

