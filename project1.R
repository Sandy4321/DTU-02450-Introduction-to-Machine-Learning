#Project 1
#Load libararies
library(ggplot2)
library(corrplot)
library(gridExtra)
library(dplyr)
library(reshape2)

#Load data set
directory <- file.path("C:","Users","gnl90","Desktop","DTU","2450 - Introduction to Machine Learning","Projects")
#directory <- file.path("C:","Users","Alp","Documents","Data Science","DTU","2450 - Introduction to Machine Learning","Projects")
dat <- read.csv(file.path(directory,"HR_data.csv"))

#Exploratory Analysis--> Changes to data set

#Summary of raw data
summary(dat)
str(dat)

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

#Make all data frame as numeric except "left" column and another data frame including"left" column
X_numeric <- cbind(X_num, X_cat_new)
X_numeric_with_left <- cbind(X_num, X_cat_new, y)

#Standardize data
X_scaled <- scale(X_numeric)

#Penalize multiple categorical variables
X_scaled[,8:17] <- X_scaled[,8:17]/sqrt(10)
X_scaled[,18:20] <- X_scaled[,18:20]/sqrt(3)
X_scaled_with_left <- cbind(X_scaled,y)

#Histograms for numerical variables
h1 <- ggplot() + geom_histogram(data = X, aes(x=satisfaction_level), col= "black", fill = "#2b8cbe", bins = 10) +
  scale_x_continuous(breaks = seq(0,1,0.1)) + xlab("Satisfaction Level") + ggtitle("Satisfaction Level Histogram")

h2 <- ggplot() + geom_histogram(data = X, aes(x=last_evaluation), col= "black", fill = "#2b8cbe", bins = 10) +
  scale_x_continuous(breaks = seq(0,1,0.1)) + xlab("Last Evaluation") + ggtitle("Last Evaluation Histogram")

h3 <- ggplot() + geom_histogram(data = X, aes(x=number_project), col= "black", fill = "#2b8cbe", bins = 6) +
  scale_x_continuous(breaks = seq(0,7,1)) + xlab("Number of Projects") + ggtitle("Number of Projects Histogram")

h4 <- ggplot() + geom_histogram(data = X, aes(x=average_montly_hours), col= "black", fill = "#2b8cbe", bins = 15) +
  scale_x_continuous(breaks = seq(0,340,10)) + xlab("Average Monthly Hours") + ggtitle("Average Monthly Hours Histogram")

h5 <- ggplot() + geom_histogram(data = X, aes(x=time_spend_company), col= "black", fill = "#2b8cbe", bins = 9) +
  scale_x_continuous(breaks = seq(0,10,1)) + xlab("Years at Company") + ggtitle("Years at Company Histogram")

grid.arrange(h1,h2,h3,h4,h5, ncol=2)

#Boxplots for numerical variables
X_scaled_df <- data.frame(X_scaled_with_left)
X_scaled_df_melt <- melt(X_scaled_df, measure.vars = 1:5)
X_scaled_df_melt <- X_scaled_df_melt[,16:18]

ggplot() + geom_boxplot(data = X_scaled_df_melt, aes(x=variable, y=value)) + ggtitle("Boxplots for Numerical Variables")
facet_labels <- c("0" = "stayed", "1" = "left")
ggplot() + geom_boxplot(data = X_scaled_df_melt, aes(x=variable, y=value)) +
  facet_grid(.~y, labeller=labeller(y = facet_labels)) + ggtitle("Boxplots for Numerical Variables for Employees that Stayed or Left")

#Plot categorical variables
sales1 <- ggplot(dat) + geom_bar(aes(x=department, fill=left), position = position_stack(reverse = T)) + 
  scale_fill_manual(values = c("#2b8cbe","#D55E00"),guide = guide_legend(reverse = T)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 15),legend.title=element_blank()) + 
  ggtitle("Barplot for number of employees per department that Stayed or Left")

sales2 <- ggplot(dat) + geom_bar(aes(x=department, fill=left), position= position_fill(reverse = T)) + 
  scale_fill_manual(values = c("#2b8cbe","#D55E00"),guide = guide_legend(reverse = T)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 15),legend.title=element_blank()) + 
  ggtitle("Barplot for percentage of employees per department that Stayed or Left") + ylab("Percentage")

grid.arrange(sales1, sales2, ncol=1)

salary1 <- ggplot(dat) + geom_bar(aes(x=salary, fill=left), position = position_stack(reverse = T)) + 
  scale_fill_manual(values = c("#2b8cbe","#D55E00"),guide = guide_legend(reverse = T)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 15),legend.title=element_blank()) + 
  ggtitle("Barplot for number of employees per Salary Level that Stayed or Left")

salary2 <- ggplot(dat) + geom_bar(aes(x=salary, fill=left), position= position_fill(reverse = T)) + 
  scale_fill_manual(values = c("#2b8cbe","#D55E00"),guide = guide_legend(reverse = T)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 15),legend.title=element_blank()) + 
  ggtitle("Barplot for percentage of employees per Salary Level that Stayed or Left") + ylab("Percentage")

grid.arrange(salary1, salary2, ncol=1)

#Binary variables
round(table(dat$left, dat$Work_accident)/N,4)*100
round(table(dat$left, dat$promotion_last_5years)/N,4)*100

#Correlation plot
observationColors <- c("#2b8cbe","#D55E00")[unclass(y+1)]
pairs(X[,1:5], bg=observationColors, pch=21)
par(xpd=TRUE)
#legend(0, 1, classNames, fill=unique(observationColors))

#Correlation of all variables with the "left variable"
cor_with_left <- data.frame(cor(X_numeric,as.numeric(dat[,7])))
cor_with_left$var <- rownames(cor_with_left)
names(cor_with_left)[1] <- "cor"
cor_with_left <- cor_with_left %>% arrange(desc(abs(cor)))
cor_with_left <- select(cor_with_left, var, cor)

#5 first chosen variables from the correlation matrix

#Satisfaction level vs left
ggplot(data = X_numeric_with_left, aes(x=satisfaction_level, y=y)) + geom_point(position = position_jitter(w=0.03, h=0.03)) + 
  geom_smooth(method = "loess")  + ggtitle("Satisfaction level vs left")

#Accidents vs left
round(table(dat$left, dat$Work_accident)/N,4)*100

#Time spend company vs left
tsc1 <- ggplot(dat) + geom_bar(aes(x=time_spend_company, fill=left), position = position_stack(reverse = T)) + 
  scale_fill_manual(values = c("#2b8cbe","#D55E00"),guide = guide_legend(reverse = T)) +
  theme(legend.title=element_blank())  + scale_x_continuous(breaks = seq(0,10,1)) + xlab("Years Spent at Company")+
  ggtitle("Time spent at company vs left")

tsc2 <- ggplot(dat) + geom_bar(aes(x=time_spend_company, fill=left), position= position_fill(reverse = T)) + 
  scale_fill_manual(values = c("#2b8cbe","#D55E00"),guide = guide_legend(reverse = T)) +
  theme(legend.title=element_blank()) + scale_x_continuous(breaks = seq(0,10,1)) + ylab("Percentage") + xlab("Years Spent at Company") + 
  ggtitle("Time spent at company vs left")

grid.arrange(tsc1, tsc2, ncol=1)

#Salaries vs left
round(table(dat$left, factor(X_numeric_with_left$sal.is.low, levels = c(0,1), labels = c("sal.not.low", "sal.low")))/N,4)*100
round(table(dat$left, factor(X_numeric_with_left$sal.is.high, levels = c(0,1), labels = c("sal.not.high", "sal.high")))/N,4)*100

#Average monthly hours vs left
ggplot(data = X_numeric_with_left, aes(x=average_montly_hours, y=y)) + geom_point(position = position_jitter(w=0.03, h=0.03)) + 
  geom_smooth(method = "loess")  + ggtitle("Average monthly hours vs left")

#remove unused variables
rm(k_list,i,j,k_names, k_list_object,low_klist, high_klist, X_cat, X_cat_new, X_num, df_klist,facet_labels)

#Singular value decomposition --> V vector is the PC vector
X_svd <- svd(X_scaled)
X_projected <- X_scaled %*% X_svd$v
colnames(X_projected) <- paste0("PC",as.character(1:dim(X_projected)[2]))

#Amount of variation explained by each principle component
pcvariance <- (X_svd$d^2)/sum(X_svd$d^2)

print(pcvariance)
print(cumsum(pcvariance))

plot(cumsum(pcvariance))

X_projected_df <- data.frame(X_projected)
X_projected_df$left <- y
X_projected_df$left <- as.factor(X_projected_df$left)
ggplot(X_projected_df) + geom_point(aes(x=PC1,y=PC2,colour = left)) + 
  theme(axis.text.x = element_text(size = 20)) +
  ggtitle("PC1 vs PC2") + 
  xlab("PC1") + 
  ylab("PC2")
