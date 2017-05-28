
library(dplyr)
library(ggplot2)
#library(plyr)
library(gridExtra)
library(doBy)
library(plotly)
library(tidyr)
library(reshape2)
## Segment Diagrams:
library(caret)
library(lubridate)
library(caret)

options(scipen=999)


###### Train set1

train1<-read.csv("./features_2iter/train_v1.csv" )

###split data
set.seed(3456)
trainIndex <- createDataPartition(train1$type_cat, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train <- train1[ trainIndex,]
test  <- train1[-trainIndex,]


#Data fro submittion
subtab<-read.csv("./features_2iter/test_v1.csv" )







#### MODEL 4
#############################3
####

# Create model with default paramters
control <- trainControl(method="repeatedcv",
                        number=5, 
                        repeats=3,
                        search = "random",
                        classProbs=TRUE,
                        savePredictions=TRUE,
                        verboseIter=TRUE)
seed <- 7

set.seed(seed)

xgb_default <- train(type_cat ~., data=train, 
                    method= "xgbTree", 
                    trControl=control,
                    tuneLength = 15,
                    metric = "LogLoss")



varImp(xgb_default)
testxgb<-cbind(test%>%select(image_name, type_cat), 
               predict(xgb_default, test ) ) 

confusionMatrix(testxgb$type_cat,testxgb$`predict(xgb_default, test)`)


submit<-cbind(predict(xgb_default, subtab, type="prob" ),subtab%>%select(image_name) ) 

saveRDS(xgb_default, "./submissions/model_m4.rds")
#saveRDS(testgbm, "./outputs/testEns_m4.rds") 
write.csv(submit, "./submissions/submit_xgb.csv", row.names = FALSE)




#######################################################################################
#######################################################################################




#### Train set3
#######################################################################################
#######################################################################################


train1<-read.csv("./features_2iter/train_v3.csv" )

###split data
set.seed(3456)
trainIndex <- createDataPartition(train1$type_cat, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train <- train1[ trainIndex,]
test  <- train1[-trainIndex,]


#Data fro submittion
subtab<-read.csv("./features_2iter/test_v3.csv" )



#### MODEL 4
#############################3
####

# Create model with default paramters
control <- trainControl(method="repeatedcv",
                        number=5, 
                        repeats=3,
                        search = "random",
                        classProbs=TRUE,
                        savePredictions=TRUE,
                        verboseIter=TRUE)
seed <- 7

set.seed(seed)

xgb_default <- train(type_cat ~., data=train, 
                     method= "xgbTree", 
                     trControl=control,
                     tuneLength = 15,
                     metric = "LogLoss")



varImp(xgb_default)
testxgb<-cbind(test%>%select(image_name, type_cat), 
               predict(xgb_default, test ) ) 

confusionMatrix(testxgb$type_cat,testxgb$`predict(xgb_default, test)`)


submit<-cbind(predict(xgb_default, subtab, type="prob" ),subtab%>%select(image_name) ) 

saveRDS(xgb_default, "./submissions/model_m4_train3.rds")
#saveRDS(testgbm, "./outputs/testEns_m4.rds") 
write.csv(submit, "./submissions/submit_xgb_train3.csv", row.names = FALSE)






#### MODEL 3
#############################3
####


# Create model with default paramters
control <- trainControl(method="repeatedcv",
                        number=5, 
                        repeats=3,
                        search = "random",
                        verboseIter=TRUE)
seed <- 7

set.seed(seed)

rf_default <- train(type_cat ~., data=train, 
                    method= "cforest", 
                    trControl=control,
                    tuneLength = 15,
                    metric = "LogLoss")




varImp(rf_default)
testrf<-cbind(test%>%select(image_name, type_cat), 
               predict(rf_default, test ) ) 

confusionMatrix(testrf$type_cat,testrf$`predict(rf_default, test)`)


submit<-cbind(predict(rf_default, subtab, type="prob" ),subtab%>%select(image_name) ) 

saveRDS(rf_default, "./submissions/model_m3_train3.rds")
#saveRDS(testgbm, "./outputs/testEns_m4.rds") 
write.csv(submit, "./submissions/submit_rf_train3.csv", row.names = FALSE)






# ###### MODEL 1
# library(gbm)
# library(caret)
# data(iris)
# fitControl <- trainControl(method="repeatedcv",
#                            number=5,
#                            repeats=10,
#                            classProbs=TRUE,
#                            savePredictions=TRUE,
#                            summaryFunction = LogLosSummary,
#                            verboseIter=TRUE)
# set.seed(825)
# gbmFit <- train(type_cat ~ ., data=train,
#                 method="gbm",
#                 metric="LogLoss",
#                 trControl=fitControl,
#                 verbose=FALSE)
# 
# 
# gbmFit
# 
# 
# #plot(gbmFit)
# varImp(gbmFit)
# testgbm<-cbind(testEns%>%select(country ,platform  ,join_date  ,source ,cohort_size,target_3_7), predict(gbmFit, testEns) ) 
# confusionMatrix(testgbm$target_3_7,testgbm$`predict(gbmFit, testEns)`)
# 
# trellis.par.set(caretTheme())
# plot(gbmFit) 
# gbmFit
# 
# saveRDS(gbmFit, "./outputs/model_m1.rds")
# saveRDS(testgbm, "./outputs/testEns_m1.rds") 
# write.csv(testgbm, "./outputs/testEns_m1.csv", row.names = FALSE)


#### MODEL 2

library(doMC)
registerDoMC(cores = 4)
## All subsequent models are then run in parallel
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=3,
                           verboseIter=TRUE)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = c(5,10,20) )

set.seed(825)
gbmFit <- train(type_cat  ~ ., data = train, 
                method = "gbm", 
                trControl = fitControl, 
                verbose = FALSE, 
                ## Now specify the exact models 
                ## to evaluate
                tuneGrid = gbmGrid)


#plot(gbmFit)
varImp(gbmFit)
testgbm<-cbind(testEns%>%select(country ,platform  ,join_date  ,source ,cohort_size,target_3_7), predict(gbmFit, testEns) ) 
confusionMatrix(testgbm$target_3_7,testgbm$`predict(gbmFit, testEns)`)

trellis.par.set(caretTheme())
plot(gbmFit) 
gbmFit

saveRDS(gbmFit, "./outputs/model_m2.rds")
saveRDS(testgbm, "./outputs/testEns_m2.rds") 
write.csv(testgbm, "./outputs/testEns_m2.csv", row.names = FALSE)



#https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
