#Project 1
#Load libararies
library(ggplot2)

#Load data set
directory <- file.path("C:","Users","gnl90","Desktop","DTU","2450 - Introduction to Machine Learning","Projects")
dat <- read.csv(file.path(directory,"HR_data.csv"))

#Exploratory Analysis--> Changes to data set---------------------------------------------------------
#Change binary variables to class factor
dat$Work_accident <- as.factor(dat$Work_accident)
dat$left <- as.factor(dat$left)
dat$promotion_last_5years <- as.factor(dat$promotion_last_5years)

#Change variable name "sales" to "department"; more precise headline
colnames(dat)[9] <- 'department'

#Representation of data + asigning to specific variable names according to class notes
attributeNames <- colnames(dat)
N <- dim(dat)[1]
M <- dim(dat)[2]

#Extract numeric variables
num_var_names <- colnames(dat)[sapply(dat[,colnames(dat)],class) %in% c("numeric","integer")]
X <- dat[,num_var_names]

#Extract categorical variables
cat_var_names <- colnames(dat)[sapply(dat[,colnames(dat)],class) %in% c("factor","character")]
X_cat <- dat[,cat_var_names]

#Summary of raw data
summary(dat)
str(dat)

#Check numerical variables distribution
for (i in 1:length(colnames(X))){
  p <- ggplot(X) + geom_density(aes(x=colnames(X)[i]))
  print(p)
}

#PCA Analysis--------------------------------------------------------------------------
X_scaled <- scale(X)
#X_scaled <- t(apply(X,1,"-",colMeans(X)))

#Singular value decomposition --> V vector is the PC vector
X_svd <- svd(X_scaled)
X_projected <- X_scaled %*% X_svd$v
colnames(X_projected) <- c("PC1","PC2","PC3","PC4","PC5")

#Amount of variation explained by each principle component
pcvariance <- (X_svd$d^2)/sum(X_svd$d^2)

plot(cumsum(pcvariance))
