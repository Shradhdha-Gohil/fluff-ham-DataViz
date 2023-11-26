#Dependencies/packages
library(mltools)
library(data.table)
library(reshape2)
library(caret)
library(caTools)
library(randomForest)
library(rpart)
library(dplyr)
library(Metrics)
library(mlbench)
library(e1071)
library(ggplot2)

#Dataset exploration
crop_production <- read.csv("D://7TH SEM//DV//Jcomp//dataset//crop_production//crop_production.csv")
head(crop_production)
str(crop_production)
unique(crop_production['Crop'])
unique(crop_production['State_Name'])

#Extracting some of the main Indian Crops
crop_production <- filter(crop_production, Crop %in% c("Rice", "Wheat", "Coffee", "Tea", "Sunflower", "Oilseeds total", "Pulses total", "Sugarcane", "Turmeric", "Maize", "Cotton"))

#Checking for NULL entries
sum(is.na(crop_production))
crop_prod_clean <- na.omit(crop_production)
sum(is.na(crop_prod_clean))

#Yield calculation
crop_prod_clean['Yield'] <- crop_prod_clean['Production']/crop_prod_clean['Area']
head(crop_prod_clean)
#write.csv(crop_prod_clean,"D://7TH SEM//DV//Jcomp//dataset//crop_production//crop_yield.csv")

#Dataset preparation for fitting in models#
model_data <- subset(crop_prod_clean, select = -c(District_Name))
head(model_data)

#One-Hot encoding on categorical features
final_data = dummyVars(" ~ .", data = model_data)
final_data = data.frame(predict(final_data, newdata = model_data))
head(final_data, 1)

#Train Test split
set.seed(3020)
sample = sample(c(TRUE, FALSE), nrow(final_data), replace = TRUE, prob = c(0.75, 0.25))
train = final_data[sample, ]
test = final_data[!sample, ]
x_train = subset(train, select = -c(Yield))
y_train = subset(train, select = c(Yield))
x_test = subset(test, select = -c(Yield))
y_test = subset(test, select = c(Yield))

#Random Forest Regressor for prediction
rf.model = randomForest(y_train$Yield ~ ., data = x_train, mtry = 3, importance = TRUE, na.action = na.omit)
rf.pred = predict(rf.model, x_test)
rf.result = postResample(pred = rf.pred, obs = y_test$Yield)
rf.result

#Decision Tree for prediction
dt.model = rpart(y_train$Yield ~ ., data = x_train)
dt.pred = predict(dt.model, x_test)
dt.result = postResample(pred = dt.pred, obs = y_test$Yield)
dt.result

#Linear Regression for prediction
lr.model = lm(y_train$Yield ~ ., data = x_train)
lr.pred = predict(lr.model, x_test)
summary(lr.model)
lr.result = postResample(pred = lr.pred, obs = y_test$Yield)
lr.result

#KNN for prediction
knn.model = knnreg(x_train, y_train$Yield)
knn.pred = predict(knn.model, x_test)
knn.result = postResample(pred = knn.pred, obs = y_test$Yield)
knn.result

#SVM Regressor for prediction
svm.model = svm(y_train$Yield ~ ., data = x_train)
svm.pred = predict(svm.model, x_test)
svm.result = postResample(pred = svm.pred, obs = y_test$Yield)
svm.result

#Scaling the dataset for better results from core regression models (Linear Regression, SVMs and KNN)
yield = final_data['Yield']
final_data_scaled = data.frame(scale(subset(final_data, select = -c(Yield)), center = TRUE, scale = TRUE))
final_data_scaled['Yield'] = yield
head(final_data_scaled, 1)

#Splitting into test and train again
set.seed(3021)
sample.scaled = sample(c(TRUE, FALSE), nrow(final_data_scaled), replace = TRUE, prob = c(0.75, 0.25))
train.scaled = final_data_scaled[sample.scaled, ]
test.scaled = final_data_scaled[!sample.scaled, ]
x_train.scaled = subset(train.scaled, select = -c(Yield))
y_train.scaled = subset(train.scaled, select = c(Yield))
x_test.scaled = subset(test.scaled, select = -c(Yield))
y_test.scaled = subset(test.scaled, select = c(Yield))

#Linear Regression when scaled
lr.model.scaled = lm(y_train.scaled$Yield ~ ., data = x_train.scaled)
lr.pred.scaled = predict(lr.model.scaled, x_test.scaled)
summary(lr.model.scaled)
lr.result.scaled = postResample(pred = lr.pred.scaled, obs = y_test.scaled$Yield)
lr.result.scaled

#KNN when scaled
knn.model.scaled = knnreg(x_train.scaled, y_train.scaled$Yield)
knn.pred.scaled = predict(knn.model.scaled, x_test.scaled)
knn.result.scaled = postResample(pred = knn.pred.scaled, obs = y_test.scaled$Yield)
knn.result.scaled

#SVM Regressor when scaled
svm.model.scaled = svm(y_train.scaled$Yield ~ ., data = x_train.scaled)
svm.pred.scaled = predict(svm.model.scaled, x_test.scaled)
svm.result.scaled = postResample(pred = svm.pred.scaled, obs = y_test.scaled$Yield)
svm.result.scaled

#Dataset creation or comparative analysis of the models
models = data.frame('Model' = c("Random Forest", "Decision Tree", "Linear Regression", "KNN", "SVMs", "Scaled Linear Regression", "Scaled KNN", "Scaled SVMs"),
                    'RMSE' = c(892.49, 168.14, 1118.56, 1133.39, 1125.14, 1147.34, 178.84, 1153.20),
                    'R2' = c(0.88, 0.98, 0.014, 0.0007, 0.0022, 0.0129, 0.98, 0.00201),
                    'MAE' = c(29.44, 16.29, 79.36, 19.84, 72.09, 82.36, 4.92, 72.74))
View(models)
write.csv(models, "D://7TH SEM//DV//Jcomp//dataset//crop_production//models.csv")

#Dataset containing actual and predicted values for all models
complete = data.frame('Actual' = y_test$Yield, 
                      'rf.pred' = rf.pred,
                      'dt.pred' = dt.pred,
                      'lr.pred' = lr.pred,
                      'knn.pred' = knn.pred,
                      'svm.pred' = svm.pred)
complete
