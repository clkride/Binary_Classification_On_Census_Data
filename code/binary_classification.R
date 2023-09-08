# Data Import
adult_census_data <- read.csv("C:/Users/abbas/OneDrive/Desktop/Practice Datasets/adult_census_data.csv", stringsAsFactors=TRUE)

# Load the dplyr package
library(dplyr)

# Lets check what our data looks like
glimpse(adult_census_data)

# Since there are no missing values, we will now do some data exploration
# Check for missing values
colSums(is.na(adult_census_data))

# Produce some numerical and graphical summaries of the data
numeric_cols <- sapply(adult_census_data, is.numeric)
pairs(adult_census_data[,numeric_cols])

# Correlation plot
# Load the corrplot package (install it if you haven't already)
library(corrplot)
# Create a correlation matrix for only the numeric columns
cor_matrix <- cor(adult_census_data[, numeric_cols])

# Create a correlation plot
corrplot(cor_matrix, method = "color")

# Summary statistics for numeric variables
summary(adult_census_data[, c("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hr_per_wk")])


# Exploring patterns in data
library(ggplot2)


# Box plot of 'Hours per Week' vs 'class'
ggplot(adult_census_data, aes(x = class, y = hr_per_wk, col = class)) + 
  geom_boxplot() + 
  theme(legend.position = "none") +  # Hide the legend
  ggtitle("Hours per Week vs class - Boxplot") 

# Box plot of 'Education Number' vs 'class'
ggplot(adult_census_data, aes(x = class, y = education_num, fill = class)) + 
  geom_boxplot() + 
  theme(legend.position = "none") +  # Hide the legend
  ggtitle("Education Number vs Class - Boxplot")

# Box plot of 'Age' vs 'class'
ggplot(adult_census_data, aes(x = class, y = age, fill = class)) + 
  geom_boxplot() + 
  theme(legend.position = "none") +  # Hide the legend
  ggtitle("Age vs class - Boxplot")

# Stacked bar plot of 'Marital Status' vs 'class'
ggplot(adult_census_data, aes(x = marital_status, fill = class)) +
  geom_bar() +
  labs(title = "Distribution of Class by Marital Status")

# Scaled Stacked Bar Plot of 'Marital Status' vs 'class'
ggplot(adult_census_data, aes(x = marital_status, fill = class)) +
  geom_bar(position = "fill") +  # Bars are on the same scale
  labs(title = "Distribution of Class by Marital Status") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

# Bar plot of 'Race' vs 'class'
ggplot(adult_census_data, aes(x = race, fill = class)) +
  geom_bar() +
  labs(title = "Distribution of Class by Race") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

# Clustered Bar Plot of 'Occupation' vs 'class'
ggplot(adult_census_data, aes(x = occupation, fill = class)) +
  geom_bar(position = "dodge") +  # Bars are clustered for each 'class'
  labs(title = "Distribution of Class by Occupation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

# Scaled Stacked Bar Plot of 'Sex' vs 'class'
ggplot(adult_census_data, aes(x = sex, fill = class)) +
  geom_bar(position = "fill") +  # Bars are on the same scale
  labs(title = "Distribution of Class by Sex")

# Splitting data into train and test sets
# Split the data into training and testing sets (70-30 split)
set.seed(123) # for reproducibility


# Rename levels to make them valid R variable names
levels(adult_census_data$class) <- make.names(levels(adult_census_data$class))

# Check the renamed levels
levels(adult_census_data$class)

# X..50K means X<=50K
# X.50K means X>50K

sample_index <- sample(1:nrow(adult_census_data), 0.7 * nrow(adult_census_data))
train_data <- adult_census_data[sample_index, ]
test_data <- adult_census_data[-sample_index, ]

nrow(train_data) / nrow(adult_census_data)


# List of columns to remove
columns_to_remove <- c("ID", "native_country", "race", "education_num","occupation")

# Remove specified columns from both train_data and test_data
train_data <- train_data[, !names(train_data) %in% columns_to_remove]
test_data <- test_data[, !names(test_data) %in% columns_to_remove]

library(caret)
library(e1071)  # For Naive Bayes
library(class)   # For k-Nearest Neighbors
library(rpart)   # For Decision Tree
install.packages("ROCR")
library(ROCR)    # For model evaluation

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)


# Logistic Regression
model_logistic <- train(class ~ ., data = train_data, method = "glm", family = "binomial", trControl = ctrl)

# Linear Discriminant Analysis (LDA)
model_lda <- train(class ~ ., data = train_data, method = "lda", trControl = ctrl)

# Naive Bayes
model_nb <- train(class ~ ., data = train_data, method = "naive_bayes", trControl = ctrl)

# k-Nearest Neighbors (KNN)
model_knn <- train(class ~ ., data = train_data, method = "knn", trControl = ctrl)

# Decision Tree
model_tree <- train(class ~ ., data = train_data, method = "rpart", trControl = ctrl)

# Evaluate the models using the re-sampling process
results <- resamples(list(
  Logistic = model_logistic,
  LDA = model_lda,
  NaiveBayes = model_nb,
  KNN = model_knn,
  DecisionTree = model_tree), test_data = test_data)

# Compare the models and plot their performance
summary(results)
bwplot(results)

summary(model_logistic)
# Our best performing model is logistic regression

# Make predictions on the test set
predictions <- predict(model_logistic, newdata = test_data)

# Create a confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$class)

# Convert the confusion matrix to a data frame
conf_matrix_df <- as.data.frame(as.table(conf_matrix))

# Create a ggplot2 plot for the confusion matrix with white text color
confusion_plot <- ggplot(data = conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq), color = "white"), vjust = 1) +  # Set text color to white
  labs(title = "Confusion Matrix",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display the confusion matrix plot
print(confusion_plot)


# Calculate specificity
specificity <- conf_matrix$byClass["Specificity"]

# Calculate recall (Sensitivity)
recall <- conf_matrix$byClass["Sensitivity"]

# Calculate precision (Positive Predictive Value)
precision <- conf_matrix$byClass["Pos Pred Value"]

# Calculate F1 score
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print key classification metrics
cat("\nSpecificity:", specificity, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("Precision (Positive Predictive Value):", precision, "\n")
cat("F1 Score:", f1_score, "\n")


# Create a ggplot2 bar plot
bar_plot <- ggplot(data = conf_matrix_df, aes(x = Reference, y = Freq, fill = Prediction)) +
  geom_bar(stat = "identity") +
  labs(title = "Confusion Matrix Bar Plot",
       x = "Actual Class",
       y = "Frequency",
       fill = "Predicted Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display the bar plot
print(bar_plot)





