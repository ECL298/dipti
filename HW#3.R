#HW 3

#7. Fit a neural network to the Default data. Use a single hidden layer with 10 units, and dropout regularization. 
#Have a look at Labs 10.9.110.9.2 for guidance. Compare the classification performance of your model with that of linear logistic regression.  

#8. From your collection of personal photographs, pick 10 images of animals (such as dogs, cats, birds, farm animals, etc.). 
#If the subject does not occupy a reasonable part of the image, then crop the image. 460 10. Deep Learning  
#Now use a pretrained image classification CNN as in Lab 10.9.4 to predict the class of each of your images, and report the probabilities for the top five predicted classes for each image.

install.packages(c("keras3"))
install.packages("keras")
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

#fit single layer neural network on Default Data
#first set up a model structure that describes the network
library(keras3)
#define model
modnn <- keras_model_sequential() |>
  layer_dense(units = 10, activation = "relu" , input_shape = ncol(x)) |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 1, activation = "sigmoid")
#complie model
compile(modnn,
        loss = "binary_crossentropy",
        optimizer = optimizer_rmsprop(), 
        metrics = list('accuracy'))
#fit model
history <- fit(modnn,
               x[-testid,], y[-testid],
               epochs=30,
               batch_size=32,
               validation_data = list(x[testid, ], y[testid]),
               verbose=1)

# Plot training history
library(ggplot2)

# Convert history metrics to a data frame
history_df <- as.data.frame(history$metrics)

# Plot training vs validation accuracy
ggplot(history_df, aes(x = seq_along(history_df$accuracy))) +
  geom_line(aes(y = accuracy, color = "Train Accuracy")) +
  geom_line(aes(y = val_accuracy, color = "Validation Accuracy")) +
  labs(x = "Epoch", y = "Accuracy") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

# Predict on test set
npred <- predict(modnn, x[testid, ])

# Convert probabilities to 0/1 classes
pred_class <- ifelse(npred > 0.5, 1, 0)

# Test accuracy
modnn_test_accuracy <- mean(pred_class == y[testid])
modnn_test_accuracy

#using pre-trained CNN
library(keras)

# 1. Path to your folder
img_dir <- "/Users/dbaral/Documents/VSCode/ECL298/animal_images"

# 2. Get image file names
image_files <- list.files(img_dir, pattern = ".jpg$", full.names = TRUE)

num_images <- length(image_files)
num_images
# 3. Create empty array
x <- array(dim = c(num_images, 224, 224, 3))

# 4. Load images
for (i in 1:num_images) {
  img <- image_load(image_files[i], target_size = c(224, 224))
  x[i,,, ] <- image_to_array(img)
}

# 5. Preprocess for ResNet
x <- imagenet_preprocess_input(x)

# 6. Load pretrained ResNet50
model <- application_resnet50(weights = "imagenet")

# 7. Predict
pred <- predict(model, x)

# 8. Decode top 5 predictions
decoded <- imagenet_decode_predictions(pred, top = 5)

names(decoded) <- basename(image_files)

print(decoded)