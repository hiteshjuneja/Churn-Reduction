
#Note : - In the below code the target variable is "Move" and if "Move" = 1, it means that customer will not move out 
#and else if Move = 2 customer will move out.

#setting The Working Directory
rm(list=ls())
setwd("D:/Data Science/Projects/Project 1")
getwd()

## installing packages

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "C50", "e1071","Rcpp","gridExtra","MASS", "gbm")
lapply(x, require, character.only = TRUE)
rm(x)



###------------------Reading Train and Test Data
train_data=read.csv("Train_data.csv",header = T,na.strings = c(" ", "", "NA"))
train_data_copy=read.csv("Train_data.csv",header = T,na.strings = c(" ", "", "NA"))
test_Data = read.csv("Test_data.csv",header=T,na.strings = c(" ","","NA"))

##--------------------Combining test_data and train_data for Exploratory Data Analysis
train_data=rbind(train_data,test_Data)
colnames(train_data)[21]='Move'

Data_Frame_train=data.frame(train_data)

###---------------------Removing the unwanted features
Data_Frame_train = subset(Data_Frame_train,select = -c(phone.number,area.code,state))

###---------------------Assigning labels to categorical variables
Data_Frame_train$Move=factor(Data_Frame_train$Move,labels = (1:length(levels(factor(Data_Frame_train$Move)))))
Data_Frame_train$voice.mail.plan=factor(Data_Frame_train$voice.mail.plan , labels = (1:length(levels(factor(Data_Frame_train$voice.mail.plan)))))
Data_Frame_train$international.plan=factor(Data_Frame_train$international.plan,labels = (1:length(levels(factor(Data_Frame_train$international.plan)))))



str(Data_Frame_train)

###----------------------Missing values Analysis
missing_val=data.frame(apply(Data_Frame_train,2,function(x){sum(is.na(x))}))

###----------------------Feature Enginerring
Data_Frame_train$total_minutes <- Data_Frame_train$total.day.minutes + Data_Frame_train$total.eve.minutes + Data_Frame_train$total.night.minutes+Data_Frame_train$total.intl.minutes
Data_Frame_train$total_calls <- Data_Frame_train$total.day.calls+Data_Frame_train$total.eve.calls+Data_Frame_train$total.night.calls+Data_Frame_train$total.intl.calls
Data_Frame_train$Avg_time <- Data_Frame_train$total_minutes/Data_Frame_train$total_calls
Data_Frame_train = Data_Frame_train[,c(1:17,19:21,18)]



### ---------------------Getting only numeric data from the Data_Frame_train set
numeric_index=sapply(Data_Frame_train, is.numeric)
numeric_data=Data_Frame_train[,numeric_index]
cnames=colnames(numeric_data)
cnames

###--------------Creating Box plot of all the numeric  data for outlier analysis

for ( i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Move"), data = subset(Data_Frame_train))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=1,
                        outlier.size=3, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(x="Move")+
           ggtitle(paste(cnames[i])))+scale_x_discrete(breaks = NULL)
}

###------------------------displaying box plot using grid Extra function
gridExtra::grid.arrange(gn1,gn2,gn3,gn4, ncol=4)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,ncol=4)
gridExtra::grid.arrange(gn9,gn10,gn11,gn12,gn13,ncol=5)
gridExtra::grid.arrange(gn14,gn15,gn16,gn17,gn18,ncol=5)



##-----Outlier Analysis----------##

##-------------Removing all the outliers---------------##
for(i in cnames){
  val = Data_Frame_train[,i][Data_Frame_train[,i] %in% boxplot.stats(Data_Frame_train[,i])$out]
  Data_Frame_train[,i][Data_Frame_train[,i] %in% val] = NA
}


##------------Calculating Missing values in data frame
missing_val_removed=data.frame(apply(Data_Frame_train,2,function(x){sum(is.na(x))}))

##---------------KNN Imputation for imputing missing values-----------##
Data_Frame_train = knnImputation(Data_Frame_train, k = 5)
sum(is.na(Data_Frame_train))



##Feature Selection##

####----Plot Correlation to check the dependency amoung numeric variables------####

corrgram(Data_Frame_train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")



## ---------------- Chi -square test for factor variable reduction
factor_index=sapply(Data_Frame_train,is.factor)
factor_data=Data_Frame_train[,factor_index]
colnames(factor_data)
factor_data_length = length(colnames(factor_data))

for(i in 1 : factor_data_length){
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Move,factor_data[,i])))
}




## ---------------- Dimesion Reduction

Data_Frame_train = subset(Data_Frame_train,select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge))


###--------------------Model Development----------------##

###----------------Seperating test and train after doing EDA

train=Data_Frame_train[1:3333,]
test = Data_Frame_train[3334:5000,]
rownames(test) <- NULL


## ------------C5.0
C50_model=C5.0(Move~.,train,trials=100,rules=TRUE)
summary(C50_model)
C50_predications=predict(C50_model,test[,-17],type="class")  

#Evaluate the performance of classfication model
confMat_C50=table(test$Move,C50_predications)
confusionMatrix(confMat_C50)


#Accuracy 94.96
##FNR 32.58

## ----------------- Random Forest

RF_model= randomForest(Move~.,train, importance = TRUE, ntree=700)
RF_Predictions = predict(RF_model, test[,-17])
confMatrix_RF = table(test$Move, RF_Predictions)
confusionMatrix(confMatrix_RF)

#Accuracy 94.6
#FNR 38.83%




###-----------Logistic Regression
logit_model = glm(Move~., data=train, family = "binomial")
logit_Predit = predict(logit_model,newdata = test , type = "response")
logit_Predit = ifelse(logit_Predit>0.5,2,1)
confMarix_LR = table(test$Move,logit_Predit)
confusionMatrix(confMarix_LR)



#Accuracy 87.76%
#FNR 83.48


# ---------------- Naive Bayes

NB_Predict = naiveBayes(Move~., train)
NB_Predictions = predict(NB_Predict, test[,-17], type ='class')
confMatrix_NB = table(test$Move, NB_Predictions)
confusionMatrix(confMatrix_NB)

#Accuarcy 89.8
#FNR 56.25





