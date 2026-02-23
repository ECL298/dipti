#HW 3

#7. Fit a neural network to the Default data. Use a single hidden layer with 10 units, and dropout regularization. 
#Have a look at Labs 10.9.110.9.2 for guidance. Compare the classification performance of your model with that of linear logistic regression.  

#8. From your collection of personal photographs, pick 10 images of animals (such as dogs, cats, birds, farm animals, etc.). 
#If the subject does not occupy a reasonable part of the image, then crop the image. 460 10. Deep Learning  
#Now use a pretrained image classification CNN as in Lab 10.9.4 to predict the class of each of your images, and report the probabilities for the top five predicted classes for each image.

install.packages(c("keras3"))
install.packages("ISLR2")
install.packages("dplyr")
library("ISLR2")
library(dplyr)

## A Single Layer Network on the Default Data

#set up the data and separate out a training and test set

# Remove missing values
Default <- na.omit(Default)

# Convert categorical variables to factors
Default$default <- as.factor(Default$default)
Default$student <- as.factor(Default$student)

# Scale numeric variables
Default_scaled <- Default %>%
  mutate(across(where(is.numeric), scale))

# Split into train/test
n <- nrow(Default_scaled)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

# Separate predictors (X) and response (y)
x <- model.matrix(default ~ . -1, data = Default_scaled)[, -1]  # remove intercept
y <- as.numeric(Default_scaled$default) - 1  # convert Yes/No to 1/0

x_train <- x[-testid, ]
x_test  <- x[testid, ]
y_train <- y[-testid]
y_test  <- y[testid]

# Convert x_train to data frame (because glm doesn't accpe a matrix directly in the formula so conver tot data frame)
x_train_df <- as.data.frame(x_train)
x_test_df  <- as.data.frame(x_test)

# Fit logistic regression
logit_model <- glm(y_train ~ ., data = x_train_df, family = binomial)

# Predicted probabilities for default = 1
logpred <- predict(logit_model, newdata = x_test_df, type = "response")
pred_class <- ifelse(logpred > 0.5, 1, 0)

log_accuracy <- mean(pred_class == y_test)
log_accuracy

##log_accuracy = 0.97