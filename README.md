<a name="readme-top"></a>
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/clkride/Binary_Classification_On_Census_Data?style=flat-square)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/m/clkride/Binary_Classification_On_Census_Data?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/clkride/Binary_Classification_On_Census_Data?style=flat-square)
![GitHub watchers](https://img.shields.io/github/watchers/clkride/Binary_Classification_On_Census_Data?style=flat-square)
![GitHub Repo stars](https://img.shields.io/github/stars/clkride/Binary_Classification_On_Census_Data?style=flat-square)
![GitHub License](https://img.shields.io/github/license/clkride/Binary_Classification_On_Census_Data?style=flat-square)
<a href="https://linkedin.com/in/abbas-singapurwala">
<img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue">
</a>

# Binary_Classification_On_Census_Data
A comprehensive analysis on different classification techniques using adult census data from US. 

## Table of Contents
- [Project Description](#project-description)
- [About Data Set](#about-data-set)
- [Data Exploration](#data-exploration)
- [Modeling Approach](#modeling-approach)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Summary](#summary)
- [Author](#author)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Description

This project employs several machine learning models, including logistic regression, Linear Discriminant Analysis (LDA), decision trees, k-nearest neighbors (KNN), and Naive Bayes. Each model will be implemented and evaluated comprehensively to compare their performance in terms of accuracy, precision, recall, F1-score, and computational efficiency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## About Data Set
### Overview
The Census Income dataset is a collection of individual records representing various attributes of individuals, including demographic, educational, and employment-related information. This dataset is often used for classification tasks, specifically for predicting whether an individual's income exceeds $50,000 (">50K") or not ("<=50K").


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Dataset Information

#### Data Columns

1. **ID**: An identifier for each individual.
2. **age**: The age of the individual.
3. **workclass**: The type of workclass the individual belongs to (e.g., State-gov, Self-emp-not-inc, Private, etc.).
4. **fnlwgt**: The final weight, which represents the number of people the census believes the entry represents.
5. **education**: The highest level of education completed by the individual.
6. **education_num**: A numerical representation of education, often equivalent to the years of education.
7. **marital_status**: The marital status of the individual.
8. **occupation**: The type of occupation the individual is engaged in.
9. **relationship**: The individual's relationship status (e.g., Husband, Not-in-family, Own-child, etc.).
10. **race**: The race of the individual.
11. **sex**: The gender of the individual.
12. **capital_gain**: The amount of capital gains the individual has.
13. **capital_loss**: The amount of capital losses the individual has.
14. **hr_per_wk**: The number of hours worked per week.
15. **native_country**: The native country of the individual.
16. **class**: The target variable, indicating whether the individual's income exceeds $50,000 (">50K") or not ("<=50K").
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Data Sample
Here's a sample of the first few records in the dataset:

| ID | age | workclass          | fnlwgt | education    | education_num | marital_status       | occupation         | relationship   | race   | sex   | capital_gain | capital_loss | hr_per_wk | native_country | class |
|----|-----|--------------------|--------|--------------|---------------|----------------------|--------------------|----------------|--------|-------|--------------|--------------|-----------|----------------|-------|
| 1  | 39  | State-gov          | 77516  | Bachelors    | 13            | Never-married        | Adm-clerical        | Not-in-family  | White  | Male  | 2174         | 0            | 40        | United-States  | <=50K |
| 2  | 50  | Self-emp-not-inc   | 83311  | Bachelors    | 13            | Married-civ-spouse  | Exec-managerial    | Husband        | White  | Male  | 0            | 0            | 13        | United-States  | <=50K |
| 3  | 38  | Private            | 215646 | HS-grad      | 9             | Divorced             | Handlers-cleaners  | Not-in-family  | White  | Male  | 0            | 0            | 40        | United-States  | <=50K |
| 4  | 53  | Private            | 234721 | 11th         | 7             | Married-civ-spouse  | Handlers-cleaners  | Husband        | Black  | Male  | 0            | 0            | 40        | United-States  | <=50K |
| 5  | 28  | Private            | 338409 | Bachelors    | 13            | Married-civ-spouse  | Prof-specialty     | Wife           | Black  | Female | 0            | 0            | 40        | Cuba           | <=50K |
| 6  | 37  | Private            | 284582 | Masters      | 14            | Married-civ-spouse  | Exec-managerial    | Wife           | White  | Female | 0            | 0            | 40        | United-States  | <=50K |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Purpose
This dataset is often used for tasks related to income prediction, demographic analysis, and employment trends. It can be used to build predictive models to classify individuals into income categories and gain insights into the factors affecting an individual's income.

### Dataset Source
The dataset is commonly used in machine learning and data science and can be found on various platforms and data repositories.

Please note that this is a simplified dataset description for use on GitHub or other platforms for data sharing and collaboration. If you have more detailed information or specific instructions for using this dataset, please provide them separately.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Exploration
In this phase of the project, I have tried to resolve common data challenges faced such as poor data quality, multicolinearity, and correlation between pair of variables. The key insights are as follows -

Insight| &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; Visualization &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;
:-------------------------|:-------------------------:
 Plot1: Duration of call v/s Subscription <br/> <br/> Higher is the duration of the last call,<br/> higher is the probability that the client will subscribe | ![alt text](https://github.com/clkride/Feature_Importance_ANN/blob/main/images/duration.png?raw=true)
  Plot2: Histogram of Duration with Subscription Overlay <br/> <br/> Subscription declines when the duration of call<br/> is close to 50 min. However, the outcome is most <br/>certainly 'yes' if the duration is close to 65 min. <br/>and 'no' when the duration exceeds 65 minutes. | ![alt text](https://github.com/clkride/Feature_Importance_ANN/blob/main/images/duration_limits.png?raw=true)
 Plot3: Months v/s Subscription <br/> <br/> Campaigns are most successful in months of - <br/> Dec, Mar, Oct, and Sep. | ![alt text](https://github.com/clkride/Feature_Importance_ANN/blob/main/images/months_vs_campaign_outcome.png?raw=true)
 Plot4: Job Type v/s Subscription <br/> <br/> Surprisingly, students and retired people are<br/> more likely to subscribe for a term deposit. | ![alt text](https://github.com/clkride/Feature_Importance_ANN/blob/main/images/job_type_vs_subscription.png?raw=true)
 Plot5: Previous Outcome v/s Current Subscription <br/> <br/> If the outcome of previous campaign was a success <br/> then the propensity of that client to subscribe <br/> the term deposit is fairly high. | ![alt text](https://github.com/clkride/Feature_Importance_ANN/blob/main/images/prev_outcome.png?raw=true)
 Plot6: Education Level v/s Subscription <br/> <br/> Illiterate people are more likely to subscribe than <br/>educated folk. Also, as the level of education <br/>increases the propensity to subscribe increases as well.| ![alt text](https://github.com/clkride/Feature_Importance_ANN/blob/main/images/edu_vs_subscription.png?raw=true)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Modeling Approach

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model Performance

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Feature Importance

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Summary

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Author
 @[Abbas S.](https://github.com/clkride)

## License
The MIT License (MIT)

Copyright (c) 2023 Abbas Singapurwala

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments
Inspiration, code snippets, etc.
* [Choose an Open Source License](https://choosealicense.com)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

