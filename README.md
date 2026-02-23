# ECL298

## This repository is to document my works from ECL298 class, winter quarter 2026 with Dr. Robert Hijmans. 


### 1. HW#1: R script that shows a form of regression and k-fold cross-validation 

-- For this HW I am using the dataset : College from ISLR package that has information related to a large number of US Colleges from the 1995 issue of US News and World Report. I am builiding a logistic regreesion model to predict whether a ceratin school is private or not using the following data from College Dataset: Apps, Top10perc, F.Undergrad, Outstate, Room.Board, Books, Personal, PhD, Terminal, S.F.Ratio, Expend student, and Grad.Rate. 

College Data Format:
A data frame with 777 observations on the following 18 variables. 
- Private: A factor with levels No and Yes indicating private or public university
- Apps: Number of applications received 
- Accept: Number of applications accepted 
- Enroll: Number of new students enrolled 
- Top10perc: Pct. new students from top 10% of H.S. class 
- Top25perc: Pct. new students from top 25% of H.S. class 
- F.Undergrad: Number of fulltime undergraduates 
- P.Undergrad: Number of parttime undergraduates 
- Outstate: Out-of-state tuition 
- Room.Board: Room and board costs 
- Books: Estimated book costs 
- Personal: Estimated personal spending 
- PhD: Pct. of faculty with Ph.D.’s 
- Terminal: Pct. of faculty with terminal degree 
- S.F.Ratio: Student/faculty ratio 
- perc.alumni: Pct. alumni who donate 
- Expend: Instructional expenditure per student 
- Grad.Rate: Graduation rate

### 2. HW#2: R script that shows RandomForest

-- For this HW I am using the same dataset used in logistic regression from HW1. 

### 3. HW#3: 

-- For this HW I am using the "Default" dataset from ISLR package. In this HW I am

Q7. Fitting a neural network to the Default data. Use a single hidden layer with 10 units, and dropout regularization.  I am comparing the classification performance of the model with that of linear logistic regression.  

Q8. From my collection of 10 animal images, I am using a pretrained image classification CNN to predict the class of each of my images, and report the probabilities for the top five predicted classes for each image.


### 4. Capstone Project 

#### A. Introduction

For this capstone project we will evaluate how the seven stress parameters unique to different rice growing stages explain yield variability in California’s Sacramento Valley using historical climate and yield data.

#### B. Data 

#### a. County Level Yield Data in California

The rice yields are from the United States Department of Agriculture (USDA) National Agricultural Statistics Service (NASS) which provides county-level crop statistics from 1980 to present. In California, rice is growin in 9 countries in Sacramento Valley: 

#### b. Climate Data

The climate data are obtained from high-resolution gridded dataset called gridMET which provides climate variables at a spatial resolution of 4km covering the contiguous United States from 1979 to present. 

#### c. Rice Cropland Layer

We use the Cropland Data Layer (CDL) which is a 30m resolution raster dataset with geographical locations of croplands in contiguous US. We use this dataset to identify the rice growing grid cells and calculate county-level average climate indices.  

#### d. Rice Phenology 

Rice growth stages are identified using Growing Degree Days Model, which has been tested to be accurate for California Rice System. Using GDD model, rice growth stages such as Booting, Flowering and Grainfill are determined. 

#### e. Temperature Indices (TI)

Temperature and Stress Indices are calculated for the three rice growth stages and for entire growth season. The following are the temperature variables and their naming convention

#### C. Statistical Modeling

We use lasso regression model for this analysis because lasso regression can mitigate the problem of multicollinearity in the variables. We train a total of 100 models over the period of 1979 to 2023 by selecting a subset of 70% of observational data and evaluating on the remaining 30%. The lasso regression model is as shown below: 

```math
Y_c(t) = F_c + \gamma_c t
+ \sum_{i=1}^{N}
\left[
\alpha \, TI_c^i(t) + \beta \, TI_c^i(t)^2
\right]
+ \varepsilon_c(t)
```

where subscripts $c$, $i$, and $t$ indicate county, temperature indices, and year, respectively.  
$Y_c(t)$ is the rice yield for county $c$ in year $t$.  
$F_c$ refers to county-level fixed effects representing average yield differences between counties.  
$\gamma_c t$ represents county-level temporal trends not explained by climate variations.  
$\alpha$ and $\beta$ are the linear and quadratic coefficients, respectively.  
$\varepsilon_c(t)$ is the error term.


