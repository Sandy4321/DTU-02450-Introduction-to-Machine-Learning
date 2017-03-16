#Project 1
#Load libararies
library(ggplot2)
library(gridExtra)
library(dplyr)
library(reshape2)
library(rpart)
library(cvTools)
library(rpart.plot)
library(FNN)

# Get functions from Toolbox
setwd("C:/Users/gnl90/Desktop/2450 - Introduction to Machine Learning/02450Toolbox_R")
source("setup.R")
setwd("C:/Users/gnl90/Desktop/2450 - Introduction to Machine Learning/Projects/Project2")

#Load data set
directory <- file.path("C:","Users","gnl90","Desktop","2450 - Introduction to Machine Learning","Projects")
#directory <- file.path("C:","Users","Alp","Documents","Data Science","DTU","2450 - Introduction to Machine Learning","Projects")
dat <- read.csv(file.path(directory,"HR_data.csv"))

######################################################################
# Exploratory Analysis - Data Preparation from Report 1###############
######################################################################

#Change binary variables to class factor
dat$Work_accident <- factor(dat$Work_accident,levels=c(0,1), labels = c("no accident", "accident"))
dat$left <- factor(dat$left, levels=c(0,1), labels = c("stayed", "left"))
dat$promotion_last_5years <- factor(dat$promotion_last_5years, levels=c(0,1), labels = c("not promoted", "promoted"))
dat$salary <- factor(dat$salary, levels =c("low", "medium","high"))

#Change variable name "sales" to "department"; more precise headline
colnames(dat)[9] <- 'department'

#Representation of data + asigning to specific variable names according to class notes
attributeNames <- colnames(dat)
attributeNames <- attributeNames[-7] 

# Extract unique class names from the "left" column
classLabels = dat[,7]
classNames = unique(classLabels)

# Extract class labels that match the class names
y = match(classLabels, classNames)
y = y*-1+2

X= dat[,attributeNames]

# Get the number of data objects, attributes, and classes
N = dim(X)[1]
M = dim(X)[2]
C = length(classNames)

#Extract numeric variables
num_var_names <- colnames(X)[sapply(X[,colnames(X)],class) %in% c("numeric","integer")]
X_num <- X[,num_var_names]

#Extract categorical variables
cat_var_names <- colnames(X)[sapply(X[,colnames(X)],class) %in% c("factor","character")]
X_cat <- dat[,cat_var_names]

#Convert categorical variables according to one out of k coding without penalization
df_klist <- list()
high_klist <- list()
low_klist <- list()
for (i in 1:dim(X_cat)[2]){
  if (length(levels(X_cat[,i])) > 2){
    k_list <- list()
    for (j in 1:length(levels(X_cat[,i]))){
      k_names <- levels(X_cat[,i])
      k_list_object <- data.frame(as.numeric(X_cat[,i] == levels(X_cat[,i])[j]))
      k_list[j] <- k_list_object
    }
    k_names <- paste0(substr(names(X_cat)[i],1,3),".","is.",k_names)
    names(k_list) <- k_names
    df_klist <- c(df_klist,k_list)
    high_klist <- c(high_klist,i)
  }else {
    X_cat[,i] <- as.numeric(X_cat[,i])-1 
    low_klist <- c(low_klist,i)
  }
}
df_klist <- data.frame(df_klist)
X_cat_new <- cbind(X_cat[,as.numeric(low_klist)], df_klist)

#Make all data frame as numeric except "left" column and another data frame including "left" column
X_numeric <- cbind(X_num, X_cat_new)
X_numeric_with_left <- cbind(X_num, X_cat_new, y)

#Standardize data
X_scaled <- scale(X_numeric)

#Penalize multiple categorical variables
X_scaled[,8:17] <- X_scaled[,8:17]/sqrt(10)
X_scaled[,18:20] <- X_scaled[,18:20]/sqrt(3)
X_scaled <- data.frame(X_scaled)

N_scaled = dim(X_scaled)[1]
M_scaled = dim(X_scaled)[2]
attributeNames_scaled <- colnames(X_scaled)

#Remove unnecessary variables
rm(df_klist,k_list_object,dat,X_cat,X_cat_new,X_num,X_numeric_with_left,i,j,k_names,low_klist,high_klist,
   directory, num_var_names, k_list)

######################################################################
# Regression #########################################################
######################################################################




######################################################################
# Classification #####################################################
######################################################################

## Crossvalidation

# Create 10-fold cross validation partition for evaluation in the outer loop
K_outer = 10
K_inner = 10
CV_outer <- cvFolds(N, K=K_outer)
# set up vectors that will store sizes of training and test sizes for outer cross validation folds
CV_outer$TrainSize_outer <- c()
CV_outer$TestSize_outer <- c()

# Initialize error variables for each model's outer cross validation fold
Error_Base_outer = rep(NA, times=K_outer)
Error_LogReg_outer = rep(NA, times=K_outer)
Error_DecTree_outer = rep(NA, times=K_outer)
Error_KNN_outer = rep(NA, times=K_outer)

# Pruning levels
prune_level = seq(from=0, to=0.05, length.out=100)
threshold = seq(from=0, to=0.75, length.out=100)
neighbours = seq(from=0, to=99, length.out=100)

# For each outer crossvalidation fold
for(k in 1:K_outer){
  print(paste('Outer crossvalidation fold ', k, '/', K_outer, sep=''))
  
  # Extract the training and test set for decision tree and logistic regression models
  X_train_outer <- X[CV_outer$which!=k, ]
  y_train_outer <- y[CV_outer$which!=k]
  X_test_outer <- X[CV_outer$which==k, ]
  y_test_outer <- y[CV_outer$which==k]
  CV_outer$TrainSize_outer[k] <- length(y_train_outer)
  CV_outer$TestSize_outer[k] <- length(y_test_outer)
  Xdatframe_train_outer <- data.frame(X_train_outer)
  colnames(Xdatframe_train_outer) <- attributeNames
  Xdatframe_test_outer <- data.frame(X_test_outer)
  colnames(Xdatframe_test_outer) <- attributeNames
  
  # Extract the training and test set for Nearest Neighbour Model
  X_train_outer_scaled <- X_scaled[CV_outer$which!=k, ]
  y_train_outer_scaled <- as.factor(y[CV_outer$which!=k])
  X_test_outer_scaled <- X_scaled[CV_outer$which==k, ]
  y_test_outer_scaled <- as.factor(y[CV_outer$which==k])
  Xdatframe_train_outer_scaled <- data.frame(X_train_outer_scaled)
  colnames(Xdatframe_train_outer_scaled) <- attributeNames_scaled
  Xdatframe_test_outer_scaled <- data.frame(X_test_outer_scaled)
  colnames(Xdatframe_test_outer_scaled) <- attributeNames_scaled
  
  # Create 10-fold crossvalidation partition for evaluation in the inner loop
  CV_inner <- cvFolds(dim(X_train_outer)[1], K=K_inner)
  
  # set up vectors that will store sizes of training and test sizes
  CV_inner$TrainSize_inner <- c()
  CV_inner$TestSize_inner <- c()
  
  # Variable for classification error
  Error_train_inner = matrix(rep(NA, times=K_inner*length(prune_level)), nrow=K_inner)
  Error_test_inner = matrix(rep(NA, times=K_inner*length(prune_level)), nrow=K_inner)
  Error_train_LogReg_inner = matrix(rep(NA, times=K_inner*length(threshold)), nrow=K_inner)
  Error_test_LogReg_inner = matrix(rep(NA, times=K_inner*length(threshold)), nrow=K_inner)
  Error_train_KNN_inner = matrix(rep(NA, times=K_inner*length(neighbours)), nrow=K_inner)
  Error_test_KNN_inner = matrix(rep(NA, times=K_inner*length(neighbours)), nrow=K_inner)
  
  # For each inner crossvalidation fold
  for(j in 1:K_inner){
    print(paste('Inner crossvalidation fold ', j, '/', K_inner, sep=''))
    
    # Extract the training and test set for decision tree and logistic regression models
    X_train_inner <- X_train_outer[CV_inner$which!=j, ]
    y_train_inner <- y_train_outer[CV_inner$which!=j]
    X_test_inner <- X_train_outer[CV_inner$which==j, ]
    y_test_inner <- y_train_outer[CV_inner$which==j]
    CV_inner$TrainSize_inner[j] <- length(y_train_inner)
    CV_inner$TestSize_inner[j] <- length(y_test_inner)
    Xdatframe_train_inner <- data.frame(X_train_inner)
    colnames(Xdatframe_train_inner) <- attributeNames
    classassignments <- classNames[y_train_inner*-1+2]
    
    # Extract the training and test set for nearest neighbour model
    X_train_inner_scaled <- X_train_outer_scaled[CV_inner$which!=j, ]
    y_train_inner_scaled <- y_train_outer_scaled[CV_inner$which!=j]
    X_test_inner_scaled <- X_train_outer_scaled[CV_inner$which==j, ]
    y_test_inner_scaled <- y_train_outer_scaled[CV_inner$which==j]
    CV_inner$TrainSize_inner[j] <- length(y_train_inner_scaled)
    CV_inner$TestSize_inner[j] <- length(y_test_inner_scaled)
    Xdatframe_train_inner_scaled <- data.frame(X_train_inner_scaled)
    colnames(Xdatframe_train_inner_scaled) <- attributeNames_scaled
    
    # construct formula to fit automatically to avoid typing in each variable name
    (fmla <- as.formula(paste("y_train_inner ~ ", paste(attributeNames, collapse= "+"))))
    
    # fit classification tree
    mytree_inner <- rpart(fmla, data=Xdatframe_train_inner,control=rpart.control(minsplit=100, minbucket=1, cp=0), 
                          parms=list(split='gini'), method="class")
    
    # fit logistic regression model
    myLogReg_inner = glm(fmla, family=binomial(link="logit"), data=Xdatframe_train_inner)
    
    Xdatframe_test_inner <- data.frame(X_test_inner)
    colnames(Xdatframe_test_inner) <- attributeNames
    Xdatframe_test_inner_scaled <- data.frame(X_test_inner_scaled)
    colnames(Xdatframe_test_inner_scaled) <- attributeNames_scaled
    
    
    # Compute classification error for each pruning level of the decision tree
    for(n in 1:length(prune_level)){ 
      mytree_pruned <- prune(mytree_inner,prune_level[n])
      predicted_classes_train_inner<- classNames[(predict(mytree_pruned, newdat=Xdatframe_train_inner, type="vector")-3)*-1]
      predicted_classes_test_inner<- classNames[(predict(mytree_pruned, newdat=Xdatframe_test_inner, type="vector")-3)*-1]
      Error_train_inner[j,n] = sum(classNames[y_train_inner*-1+2]!= predicted_classes_train_inner)
      Error_test_inner[j,n] = sum(classNames[y_test_inner*-1+2]!= predicted_classes_test_inner)
    }
    
    # Compute classification error for each threshold level of logistic regression
    for(m in 1:length(threshold)){ 
      y_est_myLogReg_train_inner = predict(myLogReg_inner, newdata=Xdatframe_train_inner, type="response")
      y_est_myLogReg_test_inner = predict(myLogReg_inner, newdata=Xdatframe_test_inner, type="response")
      Error_train_LogReg_inner[j,m] = sum((y_train_inner)!=(y_est_myLogReg_train_inner>threshold[m]))
      Error_test_LogReg_inner[j,m] = sum((y_test_inner)!=(y_est_myLogReg_test_inner>threshold[m]))
    }
    
    for(l in 1:length(neighbours)){ # For each number of neighbors
      y_est_KNN_train_inner <- knn(X_train_inner_scaled, X_train_inner_scaled,
                                   cl=y_train_inner_scaled, k = l, prob = FALSE, algorithm="kd_tree")
      y_est_KNN_test_inner <- knn(X_train_inner_scaled, X_test_inner_scaled,
                                  cl=y_train_inner_scaled, k = l, prob = FALSE, algorithm="kd_tree")
      Error_train_KNN_inner[j,l] = sum(y_train_inner_scaled!=y_est_KNN_train_inner)
      Error_test_KNN_inner[j,l] = sum(y_test_inner_scaled!=y_est_KNN_test_inner)
    }
    
  }
  
  # Plot classification error for decision tree for each outer loop
  plot(c(min(prune_level), max(prune_level)), c(min(colSums(Error_train_inner)/sum(CV_inner$TrainSize_inner),
                                                    colSums(Error_test_inner)/sum(CV_inner$TestSize_inner)),
                                                max(colSums(Error_train_inner)/sum(CV_inner$TrainSize_inner),
                                                    colSums(Error_test_inner)/sum(CV_inner$TestSize_inner))),
       main='Inner Decision tree: 10-fold crossvalidation results', xlab = 'Pruning level', ylab='Classification error', type="n")
  points(prune_level, colSums(Error_train_inner)/sum(CV_inner$TrainSize_inner), col="blue")
  points(prune_level, colSums(Error_test_inner)/sum(CV_inner$TestSize_inner), col="red")
  legend('bottomright', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
  
  # Extract optimal pruning level which is the level with minimum test error
  calculated_prune_level <- prune_level[min(which(colSums(Error_test_inner)/sum(CV_inner$TestSize_inner) == 
                                                    min(colSums(Error_test_inner)/sum(CV_inner$TestSize_inner))))]
  
  # Plot classification error for Logistic Regression for each outer loop
  plot(c(min(threshold), max(threshold)), c(min(colSums(Error_train_LogReg_inner)/sum(CV_inner$TrainSize_inner),
                                                colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner)),
                                            max(colSums(Error_train_LogReg_inner)/sum(CV_inner$TrainSize_inner),
                                                colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner))),
       main='Inner Logistic Regression: 10-fold crossvalidation results', 
       xlab = 'Threshold', ylab='Classification error', type="n")
  points(threshold, colSums(Error_train_LogReg_inner)/sum(CV_inner$TrainSize_inner), col="blue")
  points(threshold, colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner), col="red")
  legend('topright', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
  
  # Extract optimal threshold which is the level with minimum test error
  calculated_threshold <- threshold[min(which(colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner) == 
                                                min(colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner))))]
  
  # Plot classification error for nearest neighbour for each outer loop
  plot(c(min(neighbours), max(neighbours)), c(min(colSums(Error_train_KNN_inner)/sum(CV_inner$TrainSize_inner),
                                                  colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner)),
                                              max(colSums(Error_train_KNN_inner)/sum(CV_inner$TrainSize_inner),
                                                  colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner))),
       main='Inner KNN Model: 10-fold crossvalidation results', xlab = 'N-Neighbours', ylab='Classification error', type="n")
  points(neighbours, colSums(Error_train_KNN_inner)/sum(CV_inner$TrainSize_inner), col="blue")
  points(neighbours, colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner), col="red")
  legend('bottomright', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
  
  # Extract optimal number of neighbours which is the number with minimum test error
  calculated_neighbour <- neighbours[min(which(colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner) == 
                                                 min(colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner))))]
  
  # construct formula to fit automatically to avoid typing in each variable name
  (fmla <- as.formula(paste("y_train_outer ~ ", paste(attributeNames, collapse= "+"))))
  
  # Fit decision tree for outer cross validation fold using the optimal pruning level
  mytree_outer <- rpart(fmla, data=Xdatframe_train_outer,
                        control=rpart.control(minsplit=100, minbucket=1, cp=calculated_prune_level), 
                        parms=list(split='gini'), method="class")
  
  # Fit Logistic Regression for outer cross validation fold 
  myLogReg_outer = glm(fmla, family=binomial(link="logit"), data=Xdatframe_train_outer)
  
  # Store classification error for each outer cross validation fold
  Error_Base_outer[k] = sum(y_test_outer!= 1)
  Error_DecTree_outer[k] = sum(classNames[y_test_outer*-1+2] != 
                                 classNames[(predict(mytree_outer,newdat=Xdatframe_test_outer, type="vector")-3)*-1])
  Error_LogReg_outer[k] = sum(y_test_outer!=(predict(myLogReg_outer, newdata=Xdatframe_test_outer, 
                                                     type="response")>calculated_threshold))
  Error_KNN_outer[k] = sum(y_test_outer_scaled!=knn(Xdatframe_train_outer_scaled, Xdatframe_test_outer_scaled,cl=y_train_outer_scaled,
                                                    k = calculated_neighbour, prob = FALSE, algorithm="kd_tree"))
  
}

# Determine if classifiers are significantly different
errors <- data.frame(cbind(Error_Base_outer/CV_outer$TestSize_outer,
                           Error_LogReg_outer/CV_outer$TestSize_outer, 
                           Error_DecTree_outer/CV_outer$TestSize_outer,
                           Error_KNN_outer/CV_outer$TestSize_outer)*100)
colnames(errors) <- c("Base Case","Logistic regression", "Decision tree","KNN")
boxplot(errors, ylab="Error rate (%)")

testresult <- t.test(Error_LogReg_outer, Error_KNN_outer)
if(testresult$p.value < 0.05){
  print('Classifiers are significantly different');
}else{
  print('Classifiers are NOT significantly different');
}

# Plot the best model
rpart.plot(mytree_outer)



