### --------- Setting the workirg directory

rm(list=ls())
setwd("D:/Data Science/Projects/Project 2")
getwd()

### --------- Installing packages
lib_list = c("ggplot2","gridExtra","corrgram","randomForest","usdm","sampling","caret",
             "xgboost","magrittr","dplyr","rpart",
             "Matrix","mlr","parallel","parallelMap","rpart.plot")

lapply(lib_list, require,character.only=TRUE)

rm(lib_list)


### --------- Reading the data 
bike_data = read.csv("day.csv",header = T ,na.strings = c(""," ","NA"))
bike_data_copy=bike_data
str(bike_data_copy)


### --------- Removing Features "instant" , "dteday" ,"registered" and "casual" from the dataset. These features does have any relevance.
bike_data_copy=subset(bike_data_copy,select=-c(instant,dteday,registered,casual))       
dim(bike_data_copy)

### --------- Changing the datatype of target variable from integer to numeric
bike_data_copy$cnt=as.numeric(bike_data_copy$cnt)


### --------- Changing the datatype to factor and assigning the labels
for(i in 1:ncol(bike_data_copy)){
  if(class(bike_data_copy[,i]) == 'integer'){
    bike_data_copy[,i]=as.factor(bike_data_copy[,i])
    bike_data_copy[,i]= factor(bike_data_copy[,i],labels = 1:length(levels(factor(bike_data_copy[,i]))))
  }
}

str(bike_data_copy)

### --------- Missing Value Analysis
missing_value = data.frame(apply(bike_data_copy, 2, function(x){sum(is.na(x))}))


### --------- Getting all the numeric columns from the data frame
numeric_index = sapply(bike_data_copy,is.numeric)
numeric_data=bike_data_copy[,numeric_index]
cnames = colnames(numeric_data)

cnames = cnames[-5]
cnames

### --------- Creating box plot of the numeric data for outlier analysis
for ( i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bike_data_copy))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=1,
                        outlier.size=3, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(x="Count")+
           ggtitle(paste(cnames[i])))+scale_x_discrete(breaks = NULL)
}



### --------- Plotting Box Plot on the Plot window
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=4)

### --------- Creating the histogram of the features having outliers plot
hist(bike_data_copy$windspeed)
hist(bike_data_copy$hum)


### --------- Finding all the outliers in the data set and replace them with the floor and ceiling values.
for(i in cnames){
  print(i)
  percentiles = quantile(bike_data_copy[,i],c(0.75,0.25))
  q75= percentiles[[1]]
  q25= percentiles[[2]]
  iqr = q75-q25
  min <- q25 - (1.5*iqr)
  max <- q75 + (1.5*iqr)
  bike_data_copy[,i][bike_data_copy[,i] < min]= min
  bike_data_copy[,i][bike_data_copy[,i] > max]= max
}



### --------- Histogram of features after removing the outliers
hist(bike_data_copy$windspeed)
hist(bike_data_copy$hum)



### --------- Feature Selection 
### --------- Plotting Correlation plot of the numeric data
corrgram(bike_data_copy[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


### --------- Checking the correlation among all the features using variance inflation factor.

bike_data_copy_2= bike_data_copy
for(i in 1:ncol(bike_data_copy_2)){
  if(class(bike_data_copy_2[,i]) == 'factor'){
    bike_data_copy_2[,i]=as.numeric(bike_data_copy_2[,i])
  }
}

vif(bike_data_copy_2[,-12])
vifcor(bike_data_copy_2[,-12],th=0.9)


### -------- Feature Selection

bike_data_copy_rel = subset(bike_data_copy,select =-c(atemp))
str(bike_data_copy_rel)

### -------- Modeling

### -------- Spliting the data into train and test data
set.seed(1234)
train.index = sample(1:nrow(bike_data_copy_rel),0.8*nrow(bike_data_copy_rel))
train_data = bike_data_copy_rel[train.index,]
test_data = bike_data_copy_rel[-train.index,]

str(train_data)


### --------- Defining Function which will we used to find the accuracy of the model
## Mean Absolute Percentage Error
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}

## Mean Absolute Error
MAE = function(y,f){
  mean(abs(y-f))
}

## Root Mean Square Error
RMSE = function(y,f){
  sqrt(mean((y-f)^2))
}


## Accuracy
Acc = function(test_data_true, predicted_values){
  mean_abs_error = format(round(MAE(test_data_true,predicted_values),2),nsmall = 2)
  root_mean_sq_er = format(round(RMSE(test_data_true,predicted_values),2),nsmall = 2)
  Error = format(round(MAPE(test_data_true,predicted_values),2),nsmall = 2)
  Accuracy = 100 - as.numeric(Error)
  print(paste0("Mean Absolute Error : ", mean_abs_error))
  print(paste0("Mean Absolute Percentage Error : " , Error))
  print(paste0("Root Mean Square Error : ", root_mean_sq_er))
  print(paste0("Accuracy : ", Accuracy))
}



### -------- Decision Tree 

### -------- Rpart for regreesion
dt_model = rpart(cnt~.,data=train_data,method = 'anova')
rpart.plot(dt_model)
rpart.rules(dt_model)
predict_dt = predict(dt_model,test_data[,-11])
Acc(test_data[,11],predict_dt)


## Error Rate 21.98
## Accuracy 78.02


### --------- Linear Regression
lm_model = lm(cnt~.,data= train_data)
summary(lm_model)
predictions_LR = predict(lm_model,test_data[,-11])
Acc(test_data[,11], predictions_LR)

## Error Rate 20.14
## Accuracy 79.86

### --------- Random Forest Aglorithm
RF_model = randomForest(cnt ~ ., train_data, importance = TRUE, ntree = 300)
RF_Predictions = predict(RF_model, test_data[,-11])

Acc(test_data[,11], RF_Predictions)

## Error 19.94
## Accuracy 80.06

### Creating Random Forest Model using MLR library
## Hypertuning parameters of random forest model
## Create task
traintask = makeRegrTask( data = train_data, target = "cnt")
testtask = makeRegrTask( data = test_data, target = "cnt")

rf.lrn <- makeLearner("regr.randomForest")
rf.lrn$par.vals <- list(ntree = 100L, importance=TRUE)

#set 5 fold cross validation
rdesc <- makeResampleDesc("CV",iters=5L)


## set parallelbackend
parallelStartSocket(cpus = detectCores())

r <- resample(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(mae,mape,rmse), show.info = T)


getParamSet(rf.lrn)

#Set parameters space
params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 10),makeIntegerParam("nodesize",lower = 10,upper = 50))


#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 10L)

#start tuning
tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(mae,mape,rmse), 
                   par.set = params, control = ctrl, show.info = T)
tune$x


#Set hyperparameters
lrn_tune <- setHyperPars(rf.lrn,par.vals = list(mtry = 5, nodesize=10))

#train model
xgmodel_rf <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred_rf <- data.frame(predict(xgmodel_rf,testtask))
Acc(xgpred_rf$response,xgpred_rf$truth)

##Error 12.78
## Accuracy 87.22



### --------- Gradient Boosting Algorithm
## creating Spare Matrix which converts Categorical Variables to dummy variables
trainm = sparse.model.matrix(cnt~.-1,data = train_data)
train_label <- train_data[,"cnt"]
train_matrix = xgb.DMatrix(data = as.matrix(trainm),label = train_label)

testm = sparse.model.matrix(cnt~.-1,data = test_data)
test_label = test_data[,"cnt"]
test_matrix = xgb.DMatrix(data = as.matrix(testm), label = test_label)



## Defining Parameters for Xgboost
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.2, gamma=0, max_depth=6, 
               min_child_weight=1, 
               subsample=1, colsample_bytree=1)

## Cross Validation - for finding the minimum number of rounds required to find the best accuracy

xgbcv <- xgb.cv( params = params, data = train_matrix, nrounds = 100, nfold = 5, showsd = F, stratified = T, 
                 maximize = F)

## Creating watchlist
watchlist = list(train = train_matrix, test = test_matrix)

## Apply XGBoost Algorithm for train data
xgb1 <- xgb.train (params = params, data = train_matrix, nrounds = 43, watchlist = watchlist, 
                   print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "rmse")

## Predicting the values of test data using training data
xgb_predict = predict(xgb1,test_matrix)
Acc(test_data[,11],xgb_predict)

##Error 15.78
##Acuracy 84.22


## --------- XGBoost with Hyperparameter tuning
### -------- Random / Grid search procedure


## One Hot Encoding
traintask = createDummyFeatures(obj = traintask)
testtask = createDummyFeatures(obj = testtask)


## Create Learner
lrn = makeLearner("regr.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="reg:linear", eval_metric="rmse", nrounds=100L, eta=0.1)


## set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

## set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = F,iters=5L)

## search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)


mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                     par.set = params, control = ctrl, show.info = T)


mytune$y


## Set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

## train model
xgmodel <- train(learner = lrn_tune,task = traintask)

## predict model
xgpred <- data.frame(predict(xgmodel,testtask))
Acc(xgpred$response,xgpred$truth)

#Error 11.87
#Accuracy 88.13

