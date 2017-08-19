#Project3

#Part1

# Get functions from Toolbox
setwd("C:/Users/gnl90/Desktop/2450 - Introduction to Machine Learning/02450Toolbox_R")
source("setup.R")
setwd("C:/Users/gnl90/Desktop/2450 - Introduction to Machine Learning/Projects/Project3")
# setwd("C:/Users/Alp/Documents/Data Science/DTU_2450 - Introduction to Machine Learning/02450Toolbox_R")
# source("setup.R")
# setwd("C:/Users/Alp/Documents/Data Science/DTU_2450 - Introduction to Machine Learning/Projects/Project3")

#Load libraries
library(mixtools)
library(mclust)
library(cvTools)
library(FNN)

#Load data set
#directory <- file.path("C:","Users","gnl90","Desktop","2450 - Introduction to Machine Learning","Projects")
directory <- file.path("C:","Users","Alp","Documents","Data Science","DTU_2450 - Introduction to Machine Learning","Projects")
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
X_scaled_with_left <- scale(X_numeric_with_left)

#Penalize multiple categorical variables
X_scaled[,8:17] <- X_scaled[,8:17]/sqrt(10)
X_scaled[,18:20] <- X_scaled[,18:20]/sqrt(3)
X_scaled <- data.frame(X_scaled, stringsAsFactors = F)

X_scaled_with_left[,8:17] <- X_scaled_with_left[,8:17]/sqrt(10)
X_scaled_with_left[,18:20] <- X_scaled_with_left[,18:20]/sqrt(3)
X_scaled_with_left <- data.frame(X_scaled_with_left, stringsAsFactors = F)

N_scaled = dim(X_scaled)[1]
M_scaled = dim(X_scaled)[2]
attributeNames_scaled <- colnames(X_scaled)

N_scaled_with_left = dim(X_scaled_with_left)[1]
M_scaled_with_left = dim(X_scaled_with_left)[2]

#Remove unnecessary variables
rm(df_klist,k_list_object,dat,X_cat,X_cat_new,X_num,X_numeric_with_left,i,j,k_names,low_klist,high_klist,
   directory, num_var_names, k_list)

####################################################################
####################################################################
######## GMM #######################################################
####################################################################
####################################################################

Xdf <- X_scaled_with_left[sample(1:nrow(X_scaled_with_left), 3000,replace=FALSE),]
y <- Xdf[,21]>0
N = dim(Xdf)[1]

## Gaussian mixture model

# Range of K's to try
KRange = 2:100
T = length(KRange)

# Allocate variables
BIC = rep(NA, times=T)
AIC = rep(NA, times=T)
CVE = rep(0, times=T)

# Create crossvalidation partition for evaluation
NumTestSets <- 10
CV <- cvFolds(N, K=NumTestSets)
CV$NumTestSets <- NumTestSets
# set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()

# For each model order (number of clusters)
for(t in 1:T){
  # Get the current K
  K = KRange[t]
  
  # Display information
  print(paste('Fitting model for K =', K))
  
  model <- Mclust(data=Xdf, G=K) # if using the package mclust to fit the model
  
  # Get BIC and AIC
  BIC[t] = BIC(model)
  AIC[t] = AIC(model)
  
  # For each crossvalidation fold
  for(k in 1:CV$NumTestSets){
    # Extract the training and test set
    X_train <- Xdf[CV$which!=k, ]
    X_test <- Xdf[CV$which==k, ]
    
    # Fit model to training set
    model <- Mclust(data=X_train, G=K) # if using the package mclust to fit the model
    
    # Evaluation crossvalidation error
    res <- gmmposterior(model, X_test)
    NLOGL <- res$ll
    CVE[t] = CVE[t]+NLOGL
  }
}

## Plot results
par(mfrow=c(1,1), cex = 0.75)
cols <- c('blue', 'darkgreen', 'red')
miny <- min(c(BIC, AIC, 2*CVE))
maxy <- max(c(BIC, AIC, 2*CVE))
plot(c(KRange[1], KRange[length(KRange)]), c(miny, maxy), main='GMM: Number of clusters', xlab='Number of Clusters',
     ylab = " ")
points(KRange, BIC, col=cols[1])
points(KRange, AIC, col=cols[2])
points(KRange, 2*CVE, col=cols[3])
par(xpd=TRUE)
legend('topleft', legend=c('BIC', 'AIC', '-2*L'), fill=cols)

best_K <- KRange[min(which(CVE==min(CVE)))]
model <- Mclust(data=Xdf, G=6)

# Get clustering
i_gmm = model$classification

# Get cluster centers
Xc = t(model$parameters$mean)

# Plot clustering centers
clusterplot(Xdf, y, i_gmm, Xc, main='GMM: Clustering')

####################################################################
####################################################################
######## Hierarchical Clustering ###################################
####################################################################
####################################################################

# Maximum number of clusters
Maxclust = 6
#Xdf_matrix <- as.matrix(Xdf)

# Compute hierarchical clustering
hc <- hclust(dist(Xdf), method="ward.D")
#hc <- hclust(as.dist(cosine(Xdf_matrix)), method="single")

# Compute clustering by thresholding the dendrogram
i_hc <- cutree(hc, k = Maxclust)

# Plot dendrogram
plot(hc)

# Plot data
clusterplot(Xdf, y, i_hc, main='Hierarchical')


####################################################################
####################################################################
#### Clustering Validity Measures ##################################
####################################################################
####################################################################

# Maximum number of clusters
K = best_K

# Allocate variables
Rand_hc = rep(NA, times=K)
Jaccard_hc = rep(NA, times=K)
NMI_hc = rep(NA,times = K)

Rand_gmm = rep(NA, times=K)
Jaccard_gmm = rep(NA, times=K)
NMI_gmm = rep(NA,times = K)

for(k in 2:K){
  
  # Run hclust
  hcres = hclust(dist(Xdf), method="single")
  i_hc <- cutree(hcres, k = k)
  
  # Run GMM
  model <- Mclust(data=Xdf, G=k)
  i_gmm = model$classification
  
  # Compute cluster validities
  res_hc <- clusterval(y, i_hc)
  res_gmm <- clusterval(y, i_gmm)
  
  Rand_hc[k] <- res_hc$Rand
  Jaccard_hc[k] <- res_hc$Jaccard
  NMI_hc[k] <- res_hc$NMI
  Rand_gmm[k] <- res_gmm$Rand
  Jaccard_gmm[k] <- res_gmm$Jaccard
  NMI_gmm[k] <- res_gmm$NMI
  print(paste('Fitted model for K =', k))
}

par(mfrow=c(1,3), cex = 0.5)
## Plot results for Rand Index
cols <- c('blue', 'green')#, 'red', 'lightblue')
maxy <- max(c(Rand_hc[2:length(Rand_hc)], Rand_gmm[2:length(Rand_gmm)]))
miny <- min(c(Rand_hc[2:length(Rand_hc)], Rand_gmm[2:length(Rand_gmm)]))
plot(c(1,K), c(miny, maxy), type='n', main='Cluster validity for Rand Index', xlab='Number of clusters', ylab='')
lines(1:K, Rand_hc, col=cols[1])
lines(1:K, Rand_gmm, col=cols[2])
#lines(1:K, NMI, col=cols[3])
legend('bottomleft', legend=c('HC', 'GMM'), fill=cols, text.width = 5)

## Plot results for Jaccard
cols <- c('blue', 'green')#, 'red', 'lightblue')
maxy <- max(c(Jaccard_hc[2:length(Jaccard_hc)], Jaccard_gmm[2:length(Jaccard_gmm)]))
miny <- min(c(Jaccard_hc[2:length(Jaccard_hc)], Jaccard_gmm[2:length(Jaccard_gmm)]))
plot(c(1,K), c(miny, maxy), type='n', main='Cluster validity for Jaccard Similarity', xlab='Number of clusters', ylab='')
lines(1:K, Jaccard_hc, col=cols[1])
lines(1:K, Jaccard_gmm, col=cols[2])
#lines(1:K, NMI, col=cols[3])
legend('bottomleft', legend=c('HC', 'GMM'), fill=cols, text.width = 5)

## Plot results for NMI
cols <- c('blue', 'green')#, 'red', 'lightblue')
maxy <- max(c(NMI_hc[2:length(NMI_hc)], NMI_gmm[2:length(NMI_gmm)]))
miny <- min(c(NMI_hc[2:length(NMI_hc)], NMI_gmm[2:length(NMI_gmm)]))
plot(c(1,K), c(miny, maxy), type='n', main='Cluster validity for NMI', xlab='Number of clusters', ylab='')
lines(1:K, NMI_hc, col=cols[1])
lines(1:K, NMI_gmm, col=cols[2])
#lines(1:K, NMI, col=cols[3])
legend('bottomleft', legend=c('HC', 'GMM'), fill=cols, text.width = 5)

par(mfrow=c(1,1), cex = 0.5)
## Plot results for HC
cols <- c('blue', 'green', 'red')#, 'lightblue')
maxy <- max(c(Rand_hc[2:length(Rand_hc)], Jaccard_hc[2:length(Jaccard_hc)],NMI_hc[2:length(NMI_hc)]))
miny <- min(c(Rand_hc[2:length(Rand_hc)], Jaccard_hc[2:length(Jaccard_hc)],NMI_hc[2:length(NMI_hc)]))
plot(c(1,K), c(miny, maxy), type='n', main='Cluster validity for HC', xlab='Number of clusters', ylab='')
lines(1:K, Rand_hc, col=cols[1])
lines(1:K, Jaccard_hc, col=cols[2])
lines(1:K, NMI_hc, col=cols[3])
legend('bottomright', legend=c('RAND', 'JACCARD','NMI'), fill=cols, text.width = 5)

## Plot results for GMM
par(mfrow=c(1,1), cex = 0.5)
cols <- c('blue', 'green', 'red')#, 'lightblue')
maxy <- max(c(Rand_gmm[2:length(Rand_gmm)], Jaccard_gmm[2:length(Jaccard_gmm)],NMI_gmm[2:length(NMI_gmm)]))
miny <- min(c(Rand_gmm[2:length(Rand_gmm)], Jaccard_gmm[2:length(Jaccard_gmm)],NMI_gmm[2:length(NMI_gmm)]))
plot(c(1,K), c(miny, maxy), type='n', main='Cluster validity for GMM', xlab='Number of clusters', ylab='')
lines(1:K, Rand_gmm, col=cols[1])
lines(1:K, Jaccard_gmm, col=cols[2])
lines(1:K, NMI_gmm, col=cols[3])
legend('bottomleft', legend=c('RAND', 'JACCARD','NMI'), fill=cols, text.width = 5)

best_HC_Rand <- KRange[max(which(Rand_hc[2:length(Rand_hc)]==max(Rand_hc[2:length(Rand_hc)])))]
best_HC_Jaccard <- KRange[max(which(Jaccard_hc[2:length(Jaccard_hc)]==max(Jaccard_hc[2:length(Jaccard_hc)])))]
best_HC_NMI <- KRange[max(which(NMI_hc[2:length(NMI_hc)]==max(NMI_hc[2:length(NMI_hc)])))]

best_GMM_Rand <- KRange[max(which(Rand_gmm[2:length(Rand_gmm)]==max(Rand_gmm[2:length(Rand_gmm)])))]
best_GMM_Jaccard <- KRange[max(which(Jaccard_gmm[2:length(Jaccard_gmm)]==max(Jaccard_gmm[2:length(Jaccard_gmm)])))]
best_GMM_NMI <- KRange[max(which(NMI_gmm[2:length(NMI_gmm)]==max(NMI_gmm[2:length(NMI_gmm)])))]