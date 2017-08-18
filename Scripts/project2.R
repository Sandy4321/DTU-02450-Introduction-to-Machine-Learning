#Project 2
rm(list=ls())
graphics.off()
Sys.setlocale("LC_TIME", "English")

#Load libararies
library(reshape2)
library(rpart)
library(cvTools)
library(rpart.plot)
library(FNN)
library(glmnet)
library(mise)
library(neuralnet)

# Get functions from Toolbox
setwd("C:/Users/Alp/Documents/Data Science/DTU/2450 - Introduction to Machine Learning/02450Toolbox_R")
#setwd("C:/Users/gnl90/Desktop/2450 - Introduction to Machine Learning/02450Toolbox_R")
source("setup.R")
setwd("C:/Users/Alp/Documents/Data Science/DTU/2450 - Introduction to Machine Learning/Projects/Project2")
#setwd("C:/Users/gnl90/Desktop/2450 - Introduction to Machine Learning/Projects/Project2")

#Load data set
#directory <- file.path("C:","Users","gnl90","Desktop","2450 - Introduction to Machine Learning","Projects")
directory <- file.path("C:","Users","Alp","Documents","Data Science","DTU","2450 - Introduction to Machine Learning","Projects")
dat <- read.csv(file.path(directory,"HR_data.csv"))

set.seed(1807)
#take a small sample of the data set
dat <- mysample <- dat[sample(1:nrow(dat), 500,
                              replace=FALSE),]

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
attributeNames <- attributeNames[-1] 

# Extract class labels that match the class names/here satisfaction level
satisfaction = dat[,1]
X3= dat[,attributeNames]

# Get the number of data objects, attributes, and classes
N = dim(X3)[1]
M = dim(X3)[2]

#Extract numeric variables
num_var_names <- colnames(X3)[sapply(X3[,colnames(X3)],class) %in% c("numeric","integer")]
X3_num <- X3[,num_var_names]

#Extract categorical variables
cat_var_names <- colnames(X3)[sapply(X3[,colnames(X3)],class) %in% c("factor","character")]
X3_cat <- dat[,cat_var_names]

#Convert categorical variables according to one out of k coding without penalization
df_klist <- list()
high_klist <- list()
low_klist <- list()
for (i in 1:dim(X3_cat)[2]){
    if (length(levels(X3_cat[,i])) > 2){
        k_list <- list()
        for (j in 1:length(levels(X3_cat[,i]))){
            k_names <- levels(X3_cat[,i])
            k_list_object <- data.frame(as.numeric(X3_cat[,i] == levels(X3_cat[,i])[j]))
            k_list[j] <- k_list_object
        }
        k_names <- paste0(substr(names(X3_cat)[i],1,3),".","is.",k_names)
        names(k_list) <- k_names
        df_klist <- c(df_klist,k_list)
        high_klist <- c(high_klist,i)
    }else {
        X3_cat[,i] <- as.numeric(X3_cat[,i])-1 
        low_klist <- c(low_klist,i)
    }
}
df_klist <- data.frame(df_klist)
X3_cat_new <- cbind(X3_cat[,as.numeric(low_klist)], df_klist)

#Make all data frame as numeric except "left" column and another data frame including "left" column
X3_numeric <- cbind(X3_num, X3_cat_new)
X3_numeric_with_left <- cbind(X3_num, X3_cat_new, satisfaction)

#Standardize data
X3_scaled <- scale(X3_numeric)

#Penalize multiple categorical variables
X3_scaled[,8:17] <- X3_scaled[,8:17]/sqrt(10)
X3_scaled[,18:20] <- X3_scaled[,18:20]/sqrt(3)
X3_scaled <- data.frame(X3_scaled)
satisfaction_scaled <- scale(satisfaction)

N_scaled = dim(X3_scaled)[1]
M_scaled = dim(X3_scaled)[2]
attributeNames_scaled <- colnames(X3_scaled)

#Remove unnecessary variables
rm(df_klist,k_list_object,dat,X3_cat,X3_cat_new,X3_num,X3_numeric_with_left,i,j,k_names,low_klist,high_klist,
   num_var_names, k_list)

######################################################################
# Regression #########################################################
######################################################################
funLinreg <- function(X_train, y_train, X_test, y_test){
    Xr <- data.frame(X_train)
    Xtest <- data.frame(X_test)
    if(dim(as.matrix(X_train))[2]!=0){
        xnam <- paste("X", 1:dim(as.matrix(X_train))[2], sep="")
        colnames(Xr) <- xnam
        colnames(Xtest) <- xnam
        (fmla <- as.formula(paste("y_train ~ ", paste(xnam, collapse= "+"))))
    }else{
        xnam <- 1
        (fmla <- as.formula(paste("y_train ~ ", paste(xnam, collapse= "+"))))
    }
    mod = lm(fmla, data=Xr)
    preds <- predict(mod, newdata = Xtest)
    sum((y_test-preds)^2)  
}

bmplot_new <- function(rowlabels, columnlabels, mat, main='Attributes selected', xlab='Iteration', ylab=''){
    # make room for horizontal y-labels
    par(mar = c(4, 10, 4, 0) + 0.1)
    I <- dim(mat)[1]
    M <- dim(mat)[2]
    
    image(1:(I+1), 1:(M+1), mat, col=gray((32:0)/32), xaxt="n", yaxt="n", xlab=xlab, ylab=ylab, main=main)
    
    xseq <- 1:I
    yseq <- 1:M
    axis(1, at=xseq, labels=columnlabels)
    axis(2, at=yseq+diff(yseq)[1]/2,labels=FALSE)
    
    text(par("usr")[1] - 0.1, yseq+diff(yseq)[1]/2, srt = 0, adj = 1, labels = rowlabels, xpd = TRUE)
    
    for(ixseq in xseq){
        abline(v=ixseq)
    }
    for(iyseq in yseq){
        abline(h=iyseq)
    }
}

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 3 # Outer cross validation: 3 fold
NTrain = 5 # Inner cross validation: holdout
hidden_nodes = seq(from=1, to=5, length.out=5) # Number of nodes to check:3
CV <- cvFolds(N_scaled, K=K)
# set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()

# Initialize variables
Features <- matrix(rep(NA, times=K*M_scaled), nrow=K)
Error_train <- matrix(rep(NA, times=K), nrow=K)
Error_test <- matrix(rep(NA, times=K), nrow=K)
Error_train_fs <- matrix(rep(NA, times=K), nrow=K)
Error_test_fs <- matrix(rep(NA, times=K), nrow=K)

Error_REG = rep(NA, times=K)
Error_Base = rep(NA, times=K)
Error_ANN <- rep(NA, times=K)
# Parameters for neural network classifier

fmla <- as.formula(paste("satisfaction_train ~ ", paste(attributeNames_scaled, collapse= "+")))

#here does the outer loop start
# For each crossvalidation fold
for(k in 1:K){
    # Extract the training and test set
    X3_train <- as.matrix(X3_scaled[CV$which!=k, ]);
    satisfaction_train <- satisfaction_scaled[CV$which!=k];
    X3_test <- as.matrix(X3_scaled[CV$which==k, ]);
    satisfaction_test <- satisfaction_scaled[CV$which==k];
    CV$TrainSize[k] <- length(satisfaction_train)
    CV$TestSize[k] <- length(satisfaction_test)
    
    # Use 5-fold crossvalidation for sequential feature selection in the inner loop
    fsres <- forwardSelection(funLinreg, X3_train, satisfaction_train, cvK=5, 
                              stoppingCrit = "minCostImprovement",minCostImprovement=1)
    # mod <- lm(fmla, data.frame(X3_train))
    # invisible(fsres <- step(mod, direction = "backward"))
    # chosen_feats <- attributeNames_scaled %in% (names(fsres$coefficients)[2:length(names(fsres$coefficients))]) 
    
    # Save the selected features
    #Features[k,] = chosen_feats
    Features[k,] = fsres$binaryFeatsIncluded
    
    # Compute squared error with feature subset selection
    Error_train_fs[k] = funLinreg(X3_train[,Features[k,]], satisfaction_train, X3_train[,Features[k,]], satisfaction_train)
    Error_test_fs[k] = funLinreg(X3_train[,Features[k,]], satisfaction_train, X3_test[,Features[k,]], satisfaction_test)
    
    # Create 3-fold crossvalidation partition for evaluation in the inner loop
    CV_anner <- cvFolds(dim(X3_train)[1], K=NTrain)
    
    # set up vectors that will store sizes of training and test sizes
    CV_anner$TrainSize_anner <- c()
    CV_anner$TestSize_anner <- c()
    
    mse_train_anner = matrix(rep(NA, times=NTrain*length(hidden_nodes)), nrow=NTrain)
    mse_test_anner = matrix(rep(NA, times=NTrain*length(hidden_nodes)), nrow=NTrain)
    
    X3_traindf <- data.frame(X3_train)
    colnames(X3_traindf) <- attributeNames_scaled
    X3_testdf <- data.frame(X3_test)
    colnames(X3_testdf) <- attributeNames_scaled
    
    # For each inner crossvalidation fold
    for(j in 1:NTrain){
        # Extract the training and test set for ANN model
        X_train_anner <- X3_train[CV_anner$which!=j, ]
        y_train_anner <- satisfaction_train[CV_anner$which!=j]
        X_test_anner <- X3_train[CV_anner$which==j, ]
        y_test_anner <- satisfaction_train[CV_anner$which==j]
        CV_anner$TrainSize_anner[j] <- length(y_train_anner)
        CV_anner$TestSize_anner[j] <- length(y_test_anner)
        
        Xdatframe_train_anner <- data.frame(X_train_anner)
        colnames(Xdatframe_train_anner) <- attributeNames_scaled
        Xdatframe_test_anner <- data.frame(X_test_anner)
        colnames(Xdatframe_test_anner) <- attributeNames_scaled
        
        # construct formula to fit automatically to avoid typing in each variable name
        (fmla <- as.formula(paste("y_train_anner ~ ", paste(attributeNames_scaled, collapse= "+"))))
        
        # Compute regression error for each hidden node of the ANN model
        for(n in hidden_nodes){ 
            netwrk_train = neuralnet(fmla, Xdatframe_train_anner, hidden=n, stepmax = 1e+09,
                                     threshold = 0.01,act.fct='tanh',linear.output=TRUE,err.fct='sse')
            #mse_train_anner[j,n] = sum((unlist(netwrk_train$net.result)-y_train_anner)^2)
            
            computeres_anner_train <- compute(netwrk_train, Xdatframe_train_anner)
            y_train_est = unlist(computeres_anner_train$net.result)
            computeres_anner_test <- compute(netwrk_train, Xdatframe_test_anner)
            y_test_est = unlist(computeres_anner_test$net.result)
            
            mse_train_anner[j,n] = sum((y_train_est -y_train_anner)^2)
            mse_test_anner[j,n] = sum((y_test_est -y_test_anner)^2)
        }
        
    }
    #par(mfrow=c(3,1))
    # Plot regression error for ANN model for each outer loop
    plot(c(min(hidden_nodes), max(hidden_nodes)), c(min(colSums(mse_train_anner)/sum(CV_anner$TrainSize_anner),
                                                        colSums(mse_test_anner)/sum(CV_anner$TestSize_anner)),
                                                    max(colSums(mse_train_anner)/sum(CV_anner$TrainSize_anner),
                                                        colSums(mse_test_anner)/sum(CV_anner$TestSize_anner))),
         main='Inner ANN Model: 5-fold crossvalidation results', xlab = 'Number of Hidden Nodes', 
         ylab='Sum of Squares Error (SSE)', type="n")
    points(hidden_nodes, colSums(mse_train_anner)/sum(CV_anner$TrainSize_anner), col="blue")
    points(hidden_nodes, colSums(mse_test_anner)/sum(CV_anner$TestSize_anner), col="red")
    legend('bottomright', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
    
    # Extract optimal pruning level which is the level with minimum test error
    calculated_hidden_node <- hidden_nodes[min(which(colSums(mse_test_anner)/sum(CV_anner$TestSize_anner) == 
                                                         min(colSums(mse_test_anner)/sum(CV_anner$TestSize_anner))))]
    
    
    # construct formula to fit automatically to avoid typing in each variable name
    (fmla <- as.formula(paste("satisfaction_train ~ ", paste(attributeNames_scaled, collapse= "+"))))
    
    MSEBest = Inf
    # Fit ANN model for outer cross validation fold using the optimal hidden node
    netwrk = neuralnet(fmla, X3_train, hidden=calculated_hidden_node, stepmax = 1e+09,
                       threshold = 0.01,act.fct='tanh',linear.output=TRUE,err.fct='sse')
    mse <- sum((unlist(netwrk$net.result)-satisfaction_train)^2)
    if(mse<MSEBest){
        bestnet <- netwrk
        MSEBest <- mse
    }
    
    # Predict ANN model on test data
    computeres <- compute(bestnet, X3_testdf)
    satisfaction_test_est = unlist(computeres$net.result)
    
    # Compute sum of squared error for each model
    Error_REG[k] = Error_test_fs[k]
    Error_ANN[k] = sum((satisfaction_test-satisfaction_test_est)^2)
    Error_Base[k] = sum((satisfaction_test-mean(satisfaction_test))^2)
}

# Show the selected features
bmplot_new(attributeNames_scaled, 1:K, Features, xlab='Crossvalidation fold', ylab='', main='Attributes selected')

X2 <- data.frame(X3_scaled)
X2$new.col <- satisfaction_scaled

#mysample <- X2[sample(1:nrow(X2), 500,
#                      replace=FALSE),]

#y2_scaled <- mysample[,21] 

#Thereafter we regress our 'final model' on the subsample and save the residuals:
#fit2 <- lm(y2_scaled ~ mysample$last_evaluation + mysample$sal.is.high + mysample$time_spend_company + mysample$Work_accident + mysample$left + mysample$number_project + mysample$average_montly_hours + mysample$dep.is.accounting + mysample$dep.is.hr + mysample$dep.is.RandD)
#epsilon2 <- resid(fit2)

#plot the residuals against all the continous variables (here only on)
#plot(mysample$average_montly_hours, epsilon2)


# Boxplots of SSE for 3 models
errors <- data.frame(cbind(Error_Base/CV$TestSize,
                           Error_REG/CV$TestSize, 
                           Error_ANN/CV$TestSize)*100)
colnames(errors) <- c("Base Case","Linear regression", "ANN")
boxplot(errors, ylab="Error rate (%)")

#t test
Error_ANNDF <- data.frame(Error_ANN)
Error_REGDF <- data.frame(Error_REG)

testresult_2 <- t.test(Error_REGDF[,1], Error_ANNDF[,1], paired = TRUE)
if(testresult_2$p.value < 0.05){
    print('Regression models are significantly different')
}else{
    print('Regression models are NOT significantly different')
}


######################################################################
# Classification #####################################################
######################################################################

#Load data set
directory <- file.path("C:","Users","gnl90","Desktop","2450 - Introduction to Machine Learning","Projects")
#directory <- file.path("C:","Users","Alp","Documents","Data Science","DTU","2450 - Introduction to Machine Learning","Projects")
dat <- read.csv(file.path(directory,"HR_data.csv"))

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

## Crossvalidation

# Create 3-fold cross validation partition for evaluation in the outer loop
K_outer = 2
K_inner = 3
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
prune_level = seq(from=0, to=0.05, length.out=50)
lambda = seq(from=0, to=0.07, length.out=50)
neighbours = seq(from=1, to=50, length.out=50)

# For each outer crossvalidation fold
for(k in 1:K_outer){
    #print(paste('Outer crossvalidation fold ', k, '/', K_outer, sep=''))
    
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
    X_train_outer_matrix <- data.matrix(X_train_outer)
    y_train_outer_matrix <- data.matrix(as.factor(y_train_outer))
    X_test_outer_matrix <- data.matrix(X_test_outer)
    y_test_outer_matrix <- data.matrix(as.factor(y_test_outer))
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
    
    # Create 3-fold crossvalidation partition for evaluation in the inner loop
    CV_inner <- cvFolds(dim(X_train_outer)[1], K=K_inner)
    
    # set up vectors that will store sizes of training and test sizes
    CV_inner$TrainSize_inner <- c()
    CV_inner$TestSize_inner <- c()
    
    # Variable for classification error
    Error_train_inner = matrix(rep(NA, times=K_inner*length(prune_level)), nrow=K_inner)
    Error_test_inner = matrix(rep(NA, times=K_inner*length(prune_level)), nrow=K_inner)
    Error_train_LogReg_inner = matrix(rep(NA, times=K_inner*length(lambda)), nrow=K_inner)
    Error_test_LogReg_inner = matrix(rep(NA, times=K_inner*length(lambda)), nrow=K_inner)
    Error_train_KNN_inner = matrix(rep(NA, times=K_inner*length(neighbours)), nrow=K_inner)
    Error_test_KNN_inner = matrix(rep(NA, times=K_inner*length(neighbours)), nrow=K_inner)
    
    # For each inner crossvalidation fold
    for(j in 1:K_inner){
        #print(paste('Inner crossvalidation fold ', j, '/', K_inner, sep=''))
        
        # Extract the training and test set for decision tree and logistic regression models
        X_train_inner <- X_train_outer[CV_inner$which!=j, ]
        y_train_inner <- y_train_outer[CV_inner$which!=j]
        X_test_inner <- X_train_outer[CV_inner$which==j, ]
        y_test_inner <- y_train_outer[CV_inner$which==j]
        CV_inner$TrainSize_inner[j] <- length(y_train_inner)
        CV_inner$TestSize_inner[j] <- length(y_test_inner)
        Xdatframe_train_inner <- data.frame(X_train_inner)
        X_train_inner_matrix <- data.matrix(X_train_inner)
        y_train_inner_matrix <- data.matrix(as.factor(y_train_inner))
        X_test_inner_matrix <- data.matrix(X_test_inner)
        y_test_inner_matrix <- data.matrix(as.factor(y_test_inner))
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
        myLogReg_inner = glmnet(X_train_inner_matrix, y_train_inner_matrix, family="binomial")
        
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
        
        # Compute classification error for each lambda level of logistic regression
        for(m in 1:length(lambda)){ 
            y_est_myLogReg_train_inner = predict(myLogReg_inner, newx=X_train_inner_matrix, type="response", s = lambda[m])
            y_est_myLogReg_test_inner = predict(myLogReg_inner, newx=X_test_inner_matrix, type="response", s = lambda[m])
            Error_train_LogReg_inner[j,m] = sum((y_train_inner)!=(y_est_myLogReg_train_inner>0.5))
            Error_test_LogReg_inner[j,m] = sum((y_test_inner)!=(y_est_myLogReg_test_inner>0.5))
        }
        
        for(l in neighbours){ # For each number of neighbors
            y_est_KNN_train_inner <- knn(X_train_inner_scaled, X_train_inner_scaled,
                                         cl=y_train_inner_scaled, k = l, prob = FALSE, algorithm="kd_tree")
            y_est_KNN_test_inner <- knn(X_train_inner_scaled, X_test_inner_scaled,
                                        cl=y_train_inner_scaled, k = l, prob = FALSE, algorithm="kd_tree")
            Error_train_KNN_inner[j,l] = sum(y_train_inner_scaled!=y_est_KNN_train_inner)
            Error_test_KNN_inner[j,l] = sum(y_test_inner_scaled!=y_est_KNN_test_inner)
        }
        
    }
    par(mfrow=c(3,1))
    # Plot classification error for decision tree for each outer loop
    plot(c(min(prune_level), max(prune_level)), c(min(colSums(Error_train_inner)/sum(CV_inner$TrainSize_inner),
                                                      colSums(Error_test_inner)/sum(CV_inner$TestSize_inner)),
                                                  max(colSums(Error_train_inner)/sum(CV_inner$TrainSize_inner),
                                                      colSums(Error_test_inner)/sum(CV_inner$TestSize_inner))),
         main='Inner Decision tree: 3-fold crossvalidation results', xlab = 'Pruning level', ylab='Classification error', type="n")
    points(prune_level, colSums(Error_train_inner)/sum(CV_inner$TrainSize_inner), col="blue")
    points(prune_level, colSums(Error_test_inner)/sum(CV_inner$TestSize_inner), col="red")
    legend('bottomright', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
    
    # Extract optimal pruning level which is the level with minimum test error
    calculated_prune_level <- prune_level[min(which(colSums(Error_test_inner)/sum(CV_inner$TestSize_inner) == 
                                                        min(colSums(Error_test_inner)/sum(CV_inner$TestSize_inner))))]
    
    # Plot classification error for Logistic Regression for each outer loop
    plot(c(min(lambda), max(lambda)), c(min(colSums(Error_train_LogReg_inner)/sum(CV_inner$TrainSize_inner),
                                            colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner)),
                                        max(colSums(Error_train_LogReg_inner)/sum(CV_inner$TrainSize_inner),
                                            colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner))),
         main='Inner Logistic Regression: 3-fold crossvalidation results', 
         xlab = 'Lambda', ylab='Classification error', type="n")
    points(lambda, colSums(Error_train_LogReg_inner)/sum(CV_inner$TrainSize_inner), col="blue")
    points(lambda, colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner), col="red")
    legend('topleft', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
    
    # Extract optimal lambda which is the level with minimum test error
    calculated_lambda <- lambda[min(which(colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner) == 
                                              min(colSums(Error_test_LogReg_inner)/sum(CV_inner$TestSize_inner))))]
    
    # Plot classification error for nearest neighbour for each outer loop
    plot(c(min(neighbours), max(neighbours)), c(min(colSums(Error_train_KNN_inner)/sum(CV_inner$TrainSize_inner),
                                                    colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner)),
                                                max(colSums(Error_train_KNN_inner)/sum(CV_inner$TrainSize_inner),
                                                    colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner))),
         main='Inner KNN Model: 3-fold crossvalidation results', xlab = 'N-Neighbours', ylab='Classification error', type="n")
    points(neighbours, colSums(Error_train_KNN_inner)/sum(CV_inner$TrainSize_inner), col="blue")
    points(neighbours, colSums(Error_test_KNN_inner)/sum(CV_inner$TestSize_inner), col="red")
    legend('bottomright', legend=c('Inner Training error', 'Inner Test error'), fill=c("blue", "red"))
    
    mtext(paste("Outer Fold",K_outer), outer = TRUE, cex = 1.5, side = 3)
    box("outer", col="black") 
    
    
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
    myLogReg_outer = glmnet(X_train_outer_matrix, y_train_outer_matrix, family="binomial", lambda = calculated_lambda)
    
    # Store classification error for each outer cross validation fold
    Error_Base_outer[k] = sum(y_test_outer!= 1)
    Error_DecTree_outer[k] = sum(classNames[y_test_outer*-1+2] != 
                                     classNames[(predict(mytree_outer,newdat=Xdatframe_test_outer, type="vector")-3)*-1])
    Error_LogReg_outer[k] = sum(y_test_outer!=(predict(myLogReg_outer, newx=X_test_outer_matrix,type="response")>0.5))
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

testresult <- t.test(Error_DecTree_outer, Error_KNN_outer)
if(testresult$p.value < 0.05){
    print('Classifiers are significantly different');
}else{
    print('Classifiers are NOT significantly different');
}

rpart.plot(mytree_outer)



