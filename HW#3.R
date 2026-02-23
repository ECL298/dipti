#HW 3

#installing necessary packages

#install.packages("keras3")
#install.packages("keras")
#install.packages("ISLR2")
#install.packages("dplyr")
library(ISLR2)
library(keras)
library(keras3)
library(dplyr)

#Q7. Fitting a neural network to the Default data. Use a single hidden layer with 10 units, and dropout regularization. 
# I am comparing the classification performance of the model with that of linear logistic regression.

## A Single Layer Network on the Default Data

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

#Q8. From my collection of 10 animal images, I am using a pretrained image classification CNN to predict the class of each of my images, 
#and report the probabilities for the top five predicted classes for each image.

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

#8. From your collection of personal photographs, pick 10 images of animals (such as dogs, cats, birds, farm animals, etc.). 
#If the subject does not occupy a reasonable part of the image, then crop the image. 460 10. Deep Learning  
#Now use a pretrained image classification CNN as in Lab 10.9.4 
#to predict the class of each of your images, and report the probabilities for the top five predicted classes for each image.

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

# #Results
# $chimpanzee.jpg
#   class_name class_description      score
# 1  n02480855           gorilla 0.69370794
# 2  n02481823        chimpanzee 0.13819225
# 3  n02480495         orangutan 0.10969859
# 4  n02484975            guenon 0.02377968
# 5  n02492660     howler_monkey 0.02268067

# $cow.jpg
#   class_name class_description       score
# 1  n02091244      Ibizan_hound 0.314790577
# 2  n02389026            sorrel 0.303989619
# 3  n02110806           basenji 0.238275364
# 4  n02403003                ox 0.076689504
# 5  n02412080               ram 0.006122368

# $goat.jpg
#   class_name class_description      score
# 1  n02423022           gazelle 0.21499373
# 2  n02422106        hartebeest 0.19225599
# 3  n02412080               ram 0.17952815
# 4  n02417914              ibex 0.17141792
# 5  n02403003                ox 0.09084267

# $horse.jpg
#   class_name class_description       score
# 1  n02389026            sorrel 0.974793732
# 2  n02110806           basenji 0.005421566
# 3  n02091831            Saluki 0.003873989
# 4  n02106030            collie 0.002081601
# 5  n02422106        hartebeest 0.001575225

# $koala.jpg
#   class_name class_description        score
# 1  n01882714             koala 0.9969621897
# 2  n02484975            guenon 0.0009015443
# 3  n02132136        brown_bear 0.0006739973
# 4  n01877812           wallaby 0.0001872312
# 5  n01883070            wombat 0.0001795225

# $leopard.jpg
#   class_name class_description        score
# 1  n02128385           leopard 0.6361953616
# 2  n02128925            jaguar 0.3610441685
# 3  n02130308           cheetah 0.0011465268
# 4  n02128757      snow_leopard 0.0003810262
# 5  n02123159         tiger_cat 0.0002191867

# $lion.jpg
#   class_name class_description        score
# 1  n02129165              lion 0.9937840700
# 2  n02125311            cougar 0.0015788213
# 3  n01877812           wallaby 0.0009704737
# 4  n02115913             dhole 0.0006023556
# 5  n02115641             dingo 0.0005383906

# $monkey.jpg
#   class_name class_description       score
# 1  n02487347           macaque 0.921306312
# 2  n02486410            baboon 0.027033336
# 3  n02486261             patas 0.022255538
# 4  n02484975            guenon 0.015896991
# 5  n02492035          capuchin 0.004456106

# $penguine.jpg
#   class_name class_description        score
# 1  n02056570      king_penguin 9.999988e-01
# 2  n02536864              coho 4.181495e-07
# 3  n02058221         albatross 2.191723e-07
# 4  n02484975            guenon 8.836550e-08
# 5  n02071294      killer_whale 7.182981e-08

# $tiger.jpg
#   class_name class_description        score
# 1  n02129604             tiger 0.8458510637
# 2  n02123159         tiger_cat 0.1519332975
# 3  n02128925            jaguar 0.0010491563
# 4  n02127052              lynx 0.0005106464
# 5  n02128385           leopard 0.0004732902