---
layout: default
title: "Final Report"
permalink: /final/
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

## Data Preprocessing

To ensure quality inputs for our text-classification models, we implemented a preprocessing pipeline designed to reduce noise, normalize writing style differences in the data, and improve the models' abilitiy to capture maningful linguistic patterns within potentially harmful language. Our approach involved several transformation steps applied to the data before vectorization could occur.

### Text Cleaning and Normalization

We removed special characters like "@", "#", and other forms of extraneous punctuation that do not contribute to the semantic meaning of a word or phrase. Eliminating these symbols helps reduce word and sentence sparsity while maintaining the underlying meaning and linguistic signal of the piece of text.

All text was converted to lowercase to maintain consistency across word forms. Social media users tend to use different capitalization practices across posts, so it was beneficial to generate capitalization consistency to precent the model from treating capitalized and lowercase versions of the same word as distinct features.

Lastly, emojis and other Unicode expressions were removed from the data. Although some emojis can carry strong sentiment cues, keeping them present would introduce thousands of rarely repeated tokens that we were not prepared to handle with our models. So, by deleting the emoji characters, we allow characted-level n-grams and word-level context to capture a word or phrase's sentiment and meaning.

### Character-Level and Word-Level Feature Extraction

To correctly capture words with unusual morphological structures, like elongated spellings, underscores, misspellings, we incorporated character n-grams alongside traditinoal word-level TF-IDF vectors. Character n-grams are effective for social-media style text, where unconventional formatting, informal tones, and fragmented words and phrases are commonplace. 

![N-Grams]({{ '/assets/IMG_9355.png' | relative_url }})
(Example of character-level n-gram decomposition for noisy social-media text)

During our TF-IDF Vectorization, we generated:
- Word-level features (max vocab size = 20,000)
- Character n-gram features (229,201 n-grams)
- A combined feature matrix representing both granular and semantic information

The resulting shapes of our design matrices were:
- Word Features: 33,383 x 20,000
- Character Features: 33,383 x 229,201
- Combined Features: 33,383 x 249,201

This representation allows the models to learn high-level semantics, like topics and sentiment terms and low-level textual cues

### TF-IDF Vectorization

TF-IDF Vectorization was applied to both n-gram spaces to weigh tokens based on rarity and per-document frequency. This emphasizes discriminative terms while minimizing the influence of frequent or uninformative words.

The distribution of TF-IDF values demonstrated a long-tailed pattern typical of sparse text vectors, with most values clustered between 0.15 and 0.30. This aligns with accepted expectations for large and sparse vocabulary sets. 

![TF-IDF Histogram]({{ '/assets/TF-IDF Histogram.png' | relative_url }})
(Distribution of TF-IDF scores with mean and median indicators)

![TF-IDF Log Scale]({{ '/assets/TF-IDF Log Scale.png' | relative_url }})
(Log-scaled TF-IDF frequency distribution showing heavy-tail behavior)

These visualizations show the sparsity of our vectorized dataset and justify the use of models capable of handling high-dimensional sparse inputs.

## Machine Learning Algorithms Implemented

To measure the effectiveness of different modeling approaches for cyberbullying detection, we implemented three supervised learning algorithms that are widely used in text classification. Each model was selected based on its suitability for high-dimensional sparse data, interpretability, and performance in handling noisy social media text data.

### Logistic Regression

One model we used was Logistic Regression (`sklearn.linear_model.LogisticRegression`). This is a probabilistic classifier that is capable of modeling the likelihood of encountering each category of cyberbullying. This model operates by learning feature weights that shift decision boundaries according to term frequencies, which is useful for interpreting which words or phrases influence predictive outcomes the most. Logistic Regression was performed with 2000 max iterations, an LBFGS solver, L2 penalty term, and 20000+ features.

Logistic Regression provides meaningful class probabilities, which are important when analyzing classification confidence. In the context of cyberbullying detection, these probabilities allow us to hone in on harmful or offensive content, which mitigates false negatives (undetected bullying) without increasing false positives (mistaken bullying).

### Linear Support Vector Machine (SVM)

The second model we used was a Linear SVM (`sklearn.svm.LinearSVC`), a well-researched choice for large-scale text classification tasks like this one. Linear SVM models optimize a maximum-margin hyperplane and are particularly effective in high-dimensional TF-IDF feature spaces, making them ideal for cyberbullying detection, where subtle linguistic cues, word patterns, and contextual signals can be distributed across thousands of sparse features. Linear SVM is also very fast and efficient, scaling linearly with the number of training samples. Linear SVM was performed with the 10000 max iterations, L2 penalty terms, squared hinge loss, and 20000+ features.

### Complement Naive Bayes

The third model we used was a Complement Naive Bayes (`sklearn.naive_bayes.ComplementNB`), a variant of Multinomial Naive Bayes that is specialized for imbalanced text classification. Our cyberbullying dataset contains skewed term distributions, where offensive, aggresive, and targeted words appear disproportionally concentrated within certain classes. Complement Naive Bayes assumes conditional independence of features, which is helpful in estimating term weights with information from the complement of each class. It was performed with 2000 max iterations, an alpha of 0.1, and 20000+ features.

Complement Naive Bayes is also computationally simple, which makes it suitable for high-dimensional vectors, which is good when dealing with offensive language that can dominate social media  

# Results and Discussion

Ultimately, the goal of our model is to effectively flag potentially harmful tweets, as well as be able to classify them by their type of cyberbullying. In order to evaluate how effective our model is, we used the following quantitative metrics. 

**Metrics**

1. **Accuracy Score (`accuracy_score`)** – Computes the accuracy of correctly classified tweets. Accuracy will give us a general sense of our model’s performance.  

2. **F1 Score (`f1_score`)** – Balances precision and recall across all categories of tweets. This will provide us with a better evaluation across all classes compared to accuracy.  

3. **Confusion Matrix (`confusion_matrix`)** – Evaluates classification accuracy and helps us identify patterns of misclassification. This will allow us to see if categories are confused with each other.  

4. **Log Loss (`log_loss`)** – Penalizes overconfident, incorrect predictions. This allows us to evaluate the classes that may not have the best accuracy.

![Logistic Regression Class]({{ '/assets/IMG_6679.png' | relative_url }})

![Linear SVM Class]({{ '/assets/IMG_7525.png' | relative_url }})

![Complement Naive Bayes Class]({{ '/assets/IMG_6408.png' | relative_url }})

**Goals**

In terms of the metrics, our goal was to achieve at least a 75% F1 score while having relatively balanced accuracy between all categories of cyberbullying. Using the confusion matrix, we aimed to minimize misclassifications, particularly false negatives as instances of bullying that are undetected can potentially lead to harm. On the other hand, we didn’t want to over flag non-harmful tweets as cyberbullying, as this can lead to censorship issues.

## Logistic Regression Results

Our baseline logistic regression model performed well, providing us with an accuracy score of 0.83. Additionally, we achieved an F1 score of 83%, which surpassed our goal of 75%. Based on these results and our data set, our model performed very well in terms of correctly predicting outcomes. Additionally, the F1 score suggests that for most classes, the model had high precision and recall. To further verify, we used a confusion matrix. The confusion matrix suggests that we had good results for the age, ethnicity, gender, and religion classes. It performed reasonably for the non-cyberbullying and other cyberbullying classes, but the results of these two classes were not as strong as the other classes. The model seemed to sometimes have confusion between these two classes. To further evaluate the model, we used log loss to make sure that it wasn’t confidently incorrectly predicting non-cyberbullying or other cyberbullying. The log loss score of 0.41 suggests that the model is performing well.

![Logistic Regression Confusion]({{ '/assets/logistic_regression_confusion.png' | relative_url }})

## Linear SVM Results

Our linear SVM model performed slightly better on average, with an accuracy and F1 score of 82%, again passing our target threshold of 75%. This model performed worse than our logistic regression model across most metrics, with modest decreases across most classes. We can infer from this that the linear SVM model provided us with less effective decision boundaries and is potentially a worse model compared to logistic regression. The confusion matrix for this model also shows promising results for the age, ethnicity, gender, and religion classes, as well as for the main classification between cyberbullying and non-cyberbullying tweets. Through both of these models, non-cyberbullying remains the least accurate class.

![Linear SVM Confusion]({{ '/assets/LinearSVM_Confusion.png' | relative_url }})

## Complement Naive Bayes Results

Our Complement Naive Bayes model delivered solid performance, achieving an accuracy score of 0.81. The F1 score of 82% also met and exceeded our minimum threshold of 75%. These results indicate that the model is effective at capturing the underlying patterns within the dataset and is capable of producing reliable predictions overall. Based on the confusion matrix, the model performed particularly well on the age, ethnicity, gender, and religion classes, with consistent precision and recall values across these categories. Its performance on the non-cyberbullying and other cyberbullying classes was adequate, but it was slightly weaker compared to the identity-based classes. Similar to our logistic regression model, Complement Naive Bayes showed some confusion between non-cyberbullying and other cyberbullying tweets, which is expected given the similarities often found between these categories. The resulting log loss value of 0.44 indicates that the model is generally calibrated well and avoids highly confident errors. Overall, the this model performed competitively compared to the others and provides a strong baseline for probabilistic text classification.

![Complement Naive Bayes Confusion]({{ '/assets/complement_naive_bayes_confusion.png' | relative_url }})

## Model Comparison
For the comparison of our models, the logistic regression model performed the best and had the strongest results across all metrics. Additionally, it had the most reliable performance across each of the classes. In terms of accuracy, the logistic regression was at 83%, while the linear SVM was at 81% and the complement naive bayes was the lowest at 76%. The logistic regression also performed the best in terms of F1 score, outperforming the other two models. Ultimately, the logistic regression model provided us with a strong baseline, and we tried to iterate and improve on it using linear SVM. However, this did not improve our results. Additionally, the complement naive bayes performed the worst. This is because Naive Bayes has the strict assumption of conditional independence, so it has difficulty with overlapping classes. We assumed that linear SVM works better for high dimensional text data, but since the text we looked at (Tweets) have a fixed word count, this may have impacted our results. Logistic regression likely performed the best because it produces calibrated probabilities, which help with classes that are more nuanced. Additionally, it is generally better at handling class imbalance and has more flexible decision boundaries for text. Additionally, across all models, there was some confusion between the not cyberbullying and the other cyberbullying classes. This is likely due to the context of the messages mattering more in these scenarios, since they are broader classes. 

# Conclusions and Next Steps

Overall, our cyberbullying detection project successfully demonstrated that traditional machine learning models paired with TF-IDF vectorization and other preprocessing techniques can effectively classify harmful social media content across multiple categories. All three models exceeded our minimum target of a 75% F1 score, with Linear SVM achieving the strongest overall performance and the most consistent results across the identity-based classes. While all models performed well on age, ethnicity, gender, and religion-based bullying, all three encountered difficulty distinguishing between non-cyberbullying and other cyberbullying, which highlights the limitations of sparse and context-dependent text representations. 

In the future, the next steps involve improving contextual understanding and reducing classificatoin vagueness. Incorporating potential deep learning or transformer-based models would allow the system to better capture semantic nuance and long-range dependenceis that TF-IDF cannot represent. Other improvements invlude exploring dense embeddings, which implement class imbalance strategies. Finally, future work could focus on deploying a moderation tool or API to enable real-time cyberbullying detection in practical online environments.

# References
[1] U.S. Department of Health and Human Services, “What Is Cyberbullying,” stopbullying.gov, Nov. 05, 2021. https://www.stopbullying.gov/cyberbullying/what-is-it  

[2] Aljwharah Alabdulwahab, Mohd Anul Haq, and M. Alshehri, “Cyberbullying Detection using Machine Learning and Deep Learning,” International journal of advanced computer science and applications/International journal of advanced computer science & applications, vol. 14, no. 10, Jan. 2023, doi: https://doi.org/10.14569/ijacsa.2023.0141045.  

[3] Mst Shapna Akter, Hossain Shahriar, N. Ahmed, and A. Cuzzocrea, “Deep Learning Approach for Classifying the Aggressive Comments on Social Media: Machine Translated Data Vs Real Life Data,” 2022 IEEE International Conference on Big Data (Big Data), Dec. 2022, doi: https://doi.org/10.1109/bigdata55660.2022.10020249.  

[4] P. Aggarwal and R. Mahajan, “Shielding Social Media: BERT and SVM Unite for Cyberbullying Detection and Classification,” Journal of Information Systems and Informatics, vol. 6, no. 2, pp. 607–623, Jun. 2024, doi: https://doi.org/10.51519/journalisi.v6i2.692.

# Collaboration

## Final Presentation
Below is our final presentation
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vROiK7xpWklQE41BlBgmwvn8knaVaHrdfNK4soFe231_SkdrbCp9aHqGGafebGBAYYvvm-Xv_POo0lp/pubembed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

## Gantt Chart
Please see our [Gantt Chart](CS 4641 Gantt Chart.xlsx)

## Contributions

| **Name**          | **Final Contributions**                             |
|-------------------|-----------------------------------------------------|
| Lucas Paschke     | Github Page Updates, Github Repo Mangement, Google Slides creation, Conclusions and Next Steps|
| Gabe Ashkenazi    | Data Pre-Processing, Model 3 Data Sourcing, Cleaning, and implementation|
| Matthew Bressler  | Evaluate results and metrics, next steps, model comparison|
| Ivan Yakubovich   | Model 3 Data Sourcing, Cleaning, and implementation|
| Adem Poshka       | Evaluate results and metrics, next steps|
