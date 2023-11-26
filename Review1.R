#MODEL MAKING FOR REGRESSIONS AND CLASSIFICATION...
library(dplyr)
library(caTools)
library(caret)
#Data Input...
#Crop Yield prediction for 2016 year using last 51 years data for each state in India...

rice_wheat <- read.csv("dataset\\rice_wheat_yield.csv")
head(rice_wheat)
str(rice_wheat)

#Preprocessing...
rice <- rice_wheat[ , 1:8]
head(rice)
wheat <- rice_wheat[, -c(6:8)]
head(wheat)
  
#Multi-linear regression model...
#Rice yield prediction using area and production per state...
rice_2015 = filter(rice, Year == 2015)
head(rice_2015)
rice <- setdiff(rice, rice_2015)
#other_data <- rice[, c(1:5)]
#rice <- rice[, -c(1:5)]
#set.seed(3020)

#rice = filter(rice, State.Name == state)
View(rice)     
rice$State.Name
for (state in rice$State.Name){
  #print(state)
  #Model building...
  filtered <- filter(rice, State.Name == state)
  
  rice.lm <- lm(RICE.YIELD..Kg.per.ha. ~ RICE.AREA..1000.ha. + RICE.PRODUCTION..1000.tons., data = filtered)
  filtered_2015 <- filter(rice_2015, State.Name == state)
  
  yield_2015_pred = predict(rice.lm, filtered_2015[, c(6, 7)], type = "response")
  cat("For state ", state, "R-squared value is ", R2(yield_2015_pred, filtered[, 8]))
  cat("For state ", state, "RMSE value is ", RMSE(yield_2015_pred, filtered[, 8]))
  #cat("For state ", state, "R-squared value is ", R2(yield_2015_pred, filter(rice_2015[, 8], State.Name == state)))
}
#Predicting for 2015 for validating...
yield_2015_pred <- predict(rice.lm, rice_2015[, c(6, 7)], type = "response")

#Performance metrics with actual 2015 data...
R2(yield_2015_pred, rice_2015[, 8])
RMSE(yield_2015_pred, rice_2015[, 8])
