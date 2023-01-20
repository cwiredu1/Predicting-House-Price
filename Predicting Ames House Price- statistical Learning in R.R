##################### PREDICTING AMES HOUSING PRICE - MACHINE LEARNING IN R##############
rm(list=ls())
########SETTING WORK DIRECTORY###########
setwd("C:/Users/owner/Desktop/statistical learning")

#########INSTALLING ALL REQUIRED LIBRARIES ###############
install.packages(c('ggplot2','ggthemes','scales','dplyr','mice',
                   'randomForest','data.table','gridExtra',
                   'corrplot','GGally','e1071','reshape2','lares',
                   'gbm','MASS','caret'))


library('ggplot2')
library('ggthemes')
library('scales')
library('dplyr')
library('mice')
library('randomForest')
library('data.table')
library('gridExtra')
library('corrplot')
library('GGally')
library('e1071')
library('reshape2')
library('lares')
library('gbm')
library('MASS')
library('caret')

###########LOADING DATASET############
train = read.csv("train_reg_features-1.csv")
test = read.csv("test_reg_features-1.csv")

####DATSET DIMENSION###
dim(train)
dim(test)

########## EDA ON TARGET VARIABLE #########
### Summary of target variable
summary(train$SalePrice)

### Visual view of target variable
ggplot(train, mapping = aes(x = SalePrice)) + geom_histogram(color = "black", fill = "blue")

### Target variable with the density plot
g = train$SalePrice
m<-mean(g)
std<-sqrt(var(g))
hist(g, density=20, breaks=20, prob=TRUE,
     main="normal curve over histogram")
curve(dnorm(x, mean=m, sd=std),
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

##SalePrice is not normally distributed, hence needs to be corrected
skewness(train$SalePrice)
kurtosis(train$SalePrice)


########Checking categorical columns##########
# names(train)[sapply(train, typeof) == "character"]
sum(sapply(train[,], typeof) == "character")

########Checking numeric columns##############
# names(train)[unlist(lapply(train, is.numeric))]
sum(sapply(train[,], typeof) != "character")


#########Splitting data into categorical and numerical variables#########
cat_var <- names(train)[which(sapply(train, is.character))]
# cat_car <- c(cat_var, 'BedroomAbvGr', 'HalfBath', ' KitchenAbvGr','BsmtFullBath', 'BsmtHalfBath', 'MSS
numeric_var <- names(train)[which(sapply(train, is.numeric))]


##########Creating one training dataset with categorical variable and one with numeric variable. #######
######## For Purposes of Data Visualization #####
train1_cat<-train[cat_var]
train1_num<-train[numeric_var]


######### Correlation ########
correlations <- cor(na.omit(train1_num[,-1]))
head(correlations, 2)

#########Heatmap for correlations#########
melted_correlations = melt(correlations)
head(melted_correlations, 2)

ggplot(data = melted_correlations, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.4,
                                   size = 12, hjust = 0.2))+ theme(aspect.ratio = 1) +
  coord_fixed()

#  Correlation Too clustered to make sense of, hence  Lets consider the top 10 correlated variables
# top 10
corr_cross(train, # name of dataset
           max_pvalue = 0.05, # display only significant correlations (at 5% level)
           top = 10 # display top 10 couples of variables (by correlation coefficient)
)



# Top 10 variables that correlate with SalePrice
top_10 = corr_var(train, # name of dataset
                  SalePrice, # name of variable to focus on
                  top = 10 # display top 5 correlations
)
top_10
##From the above, Blue indicate positive correlation and red indicate negative correlations

#Let’s take a look at how each relates to Sale Price and do some pre-cleaning on each feature if necessary.
# Overall Quality vs Sale Price
# unique(train$OverallQual)
ggplot(train, mapping = aes(x = factor(OverallQual), y = SalePrice)) + geom_boxplot()
## This  makes sense. People pay more for better quality.

# Living Area vs Sale Price
ggplot(train, mapping = aes(x = GrLivArea, y = SalePrice)) + geom_point() + geom_smooth(method = "lm")
##It makes sense that people would pay for the more living area. But it doesn’t make sense is to pay less
#for large area as in the two datapoints in the bottom-right of the plot. We need to take care of that by
#removing those points manually.

# Removing outliers manually (The two points in the bottom right)
train = train[train$GrLivArea<=4500,]
ggplot(train, mapping = aes(x = GrLivArea, y = SalePrice)) + geom_point() + geom_smooth(method = "lm")


# Garage Area vs Sale Price
unique(train$GarageCars)
ggplot(train, mapping = aes(x = factor(GarageCars), y = SalePrice)) + geom_boxplot()

##4-car garages result in less Sale Price? That doesn’t make much sense. Let’s remove those outliers.
train = train[train$GarageCars < 4, ]
ggplot(train, mapping = aes(x = factor(GarageCars), y = SalePrice)) + geom_boxplot()

# Garage Area vs Sale Price
ggplot(train, mapping = aes(x = GarageArea , y = SalePrice)) + geom_point() + geom_smooth(method = "lm")

##Again, the two data points at the bottom does not make sense
train = filter(train, GarageArea < 1240)
ggplot(train, mapping = aes(x = GarageArea , y = SalePrice)) + geom_point() + geom_smooth(method = "lm")

# Basement Area vs Sale Price
ggplot(train, mapping = aes(x = TotalBsmtSF , y = SalePrice)) + geom_point() + geom_smooth(method = "lm")


# First Floor Area vs Sale Price
ggplot(train, mapping = aes(x = X1stFlrSF, y = SalePrice)) + geom_point() + geom_smooth(method = "lm")

# # ExterQual vs Sale Price
ggplot(train, mapping = aes(x = factor(ExterQual), y = SalePrice)) + geom_boxplot()
#Some how this also make sense. Thus, houses with excellent external quality are expensive whereas houses
#with fair quality material are less expensive.

# ExterQual vs Sale Price
ggplot(train, mapping = aes(x = factor(FullBath), y = SalePrice)) + geom_boxplot()
#It does not make sense to me that, houses with zero full bath are expensive than houses with one full bath.
#This will be an outlier and must be removed from the data set.

train = train[train$FullBath > 0 , ]
ggplot(train, mapping = aes(x = factor(FullBath), y = SalePrice)) + geom_boxplot()

#BsmtQual and SalePrice
ggplot(train, mapping = aes(x = factor(BsmtQual), y = SalePrice)) + geom_boxplot()
#This makes sense as well.

# Total Rooms vs Sale Price
ggplot(train, mapping = aes(x = factor(TotRmsAbvGrd), y = SalePrice)) + geom_boxplot()+ scale_y_discrete()

# table(train$TotRmsAbvGrd)
It appears like houses with more than 11 rooms cost less than those with 11 rooms. 12 rooms and more have
lower median compared to those with 11 rooms. Thought of removing atleast houses with 14 rooms but its
only one house which might not make any serious difference


#########Missing Values#########
# ntrain = dim(train)[1]
# ntest = dim(test)[1]
# y_trian = train$SalePrice

# all_data = bind_rows(train,test)
# all_data$SalePrice = NULL
# head(all_data,2)

test$SalePrice <- NA
train$isTrain <- 1
test$isTrain <- 0
all_data <- rbind(train,test)
head(all_data,2)

###Getmode###
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
dim(train)

dim(test)

dim(all_data)

Missing_indices <- sapply(all_data,function(x) sum(is.na(x)))/nrow(all_data)

Missing_Summary <- data.frame(index = names(all_data),Missing_Values=Missing_indices)

Missing_Summary[order(Missing_Summary$Missing_Values > 0, decreasing = TRUE),]

sum(is.na(all_data))/ (dim(all_data)[1] *dim(all_data)[2]) *100 ### percentage of missing cases

all_data_na = Missing_Summary[order(Missing_Summary$Missing_Values, decreasing = TRUE), ]
all_data_na[1:5, ]

########Imputing missing values#######
head(all_data, 2)

#PoolQC
#Changing NA in PoolQC to None
all_data$PoolQC1 <- as.character(all_data$PoolQC)
all_data$PoolQC1[which(is.na(all_data$PoolQC))] <- "None"
all_data$PoolQC <- as.factor(all_data$PoolQC1)
all_data <- subset(all_data,select = -PoolQC1)
sum(is.na(all_data$PoolQC))

#MiscFeature
all_data$MiscFeature1 <- as.character(all_data$MiscFeature)
all_data$MiscFeature1[which(is.na(all_data$MiscFeature))] <- "None"
all_data$MiscFeature <- as.factor(all_data$MiscFeature1)
all_data <- subset(all_data,select = -MiscFeature1)
sum(is.na(all_data$MiscFeature))

#Alley
all_data$Alley1 <- as.character(all_data$Alley)
all_data$Alley1[which(is.na(all_data$Alley))] <- "None"
all_data$Alley <- as.factor(all_data$Alley1)
all_data <- subset(all_data,select = -Alley1)
sum(is.na(all_data$Alley))

#Fence
all_data$Fence1 <- as.character(all_data$Fence)
all_data$Fence1[which(is.na(all_data$Fence))] <- "None"
all_data$Fence <- as.factor(all_data$Fence1)
all_data <- subset(all_data,select = -Fence1)
sum(is.na(all_data$Fence))

#FireplaceQu
all_data$FireplaceQu1 <- as.character(all_data$FireplaceQu)
all_data$FireplaceQu1[which(is.na(all_data$FireplaceQu))] <- "None"
all_data$FireplaceQu <- as.factor(all_data$FireplaceQu1)
all_data <- subset(all_data,select = -FireplaceQu1)
sum(is.na(all_data$FireplaceQu))

unique(all_data$FireplaceQu)

#LotFrontage
all_data$LotFrontage[which(is.na(all_data$LotFrontage))] <- median(all_data$LotFrontage,na.rm = T)
sum(is.na(all_data$LotFrontage))

#GarageType
all_data$GarageType1 <- as.character(all_data$GarageType)
all_data$GarageType1[which(is.na(all_data$GarageType))] <- "None"
all_data$GarageType <- as.factor(all_data$GarageType1)
all_data <- subset(all_data,select = -GarageType1)
sum(is.na(all_data$GarageType))

#GarageFinish
all_data$GarageFinish1 <- as.character(all_data$GarageFinish)
all_data$GarageFinish1[which(is.na(all_data$GarageFinish))] <- "None"
all_data$GarageFinish <- as.factor(all_data$GarageFinish1)
all_data <- subset(all_data,select = -GarageFinish1)
sum(is.na(all_data$GarageFinish))

#GarageQual
all_data$GarageQual1 <- as.character(all_data$GarageQual)
all_data$GarageQual1[which(is.na(all_data$GarageQual))] <- "None"
all_data$GarageQual <- as.factor(all_data$GarageQual1)
all_data <- subset(all_data,select = -GarageQual1)

#GarageCond
all_data$GarageCond1 <- as.character(all_data$GarageCond)
all_data$GarageCond1[which(is.na(all_data$GarageCond))] <- "None"
all_data$GarageCond <- as.factor(all_data$GarageCond1)
all_data <- subset(all_data,select = -GarageCond1)

unique(all_data$GarageCond)

#GarageYrBlt
all_data$GarageYrBlt[which(is.na(all_data$GarageYrBlt))] <- 0
sum(is.na(all_data$GarageYrBlt))
sum(is.na(all_data$GarageCond))

# head(all_data, 2)
unique(all_data$GarageYrBlt)

#GarageArea
all_data$GarageArea[which(is.na(all_data$GarageArea))] <- 0
#GarageCars
all_data$GarageCars[which(is.na(all_data$GarageCars))] <- 0
#BsmtFinSF1
all_data$BsmtFinSF1[which(is.na(all_data$BsmtFinSF1))] <- 0
#BsmtFinSF2
all_data$BsmtFinSF2[which(is.na(all_data$BsmtFinSF2))] <- 0
#BsmtUnfSF
all_data$BsmtUnfSF[which(is.na(all_data$BsmtUnfSF))] <- 0
#TotalBsmtSF
all_data$TotalBsmtSF[which(is.na(all_data$TotalBsmtSF))] <- 0
#BsmtFullBath
all_data$BsmtFullBath[which(is.na(all_data$BsmtFullBath))] <- 0
#BsmtHalfBath
all_data$BsmtHalfBath[which(is.na(all_data$BsmtHalfBath))] <- 0
#BsmtQual
all_data$BsmtQual1 <- as.character(all_data$BsmtQual)
all_data$BsmtQual1[which(is.na(all_data$BsmtQual))] <- "None"
all_data$BsmtQual <- as.factor(all_data$BsmtQual1)
all_data <- subset(all_data,select = -BsmtQual1)
#BsmtCond
all_data$BsmtCond1 <- as.character(all_data$BsmtCond)
all_data$BsmtCond1[which(is.na(all_data$BsmtCond))] <- "None"
all_data$BsmtCond <- as.factor(all_data$BsmtCond1)
all_data <- subset(all_data,select = -BsmtCond1)

#BsmtExposure
all_data$BsmtExposure1 <- as.character(all_data$BsmtExposure)
all_data$BsmtExposure1[which(is.na(all_data$BsmtExposure))] <- "None"
all_data$BsmtExposure <- as.factor(all_data$BsmtExposure1)
all_data <- subset(all_data, select = -BsmtExposure1)

#BsmtFinType1
all_data$BsmtFinType11 <- as.character(all_data$BsmtFinType1)
all_data$BsmtFinType11[which(is.na(all_data$BsmtFinType1))] <- "None"
all_data$BsmtFinType1 <- as.factor(all_data$BsmtFinType11)
all_data <- subset(all_data,select = -BsmtFinType11)

#BsmtFinType2
all_data$BsmtFinType21 <- as.character(all_data$BsmtFinType2)
all_data$BsmtFinType21[which(is.na(all_data$BsmtFinType2))] <- "None"
all_data$BsmtFinType2 <- as.factor(all_data$BsmtFinType21)
all_data <- subset(all_data,select = -BsmtFinType21)
#MasVnrType

all_data$MasVnrType1 <- as.character(all_data$MasVnrType)
all_data$MasVnrType1[which(is.na(all_data$MasVnrType))] <- "None"
all_data$MasVnrType <- as.factor(all_data$MasVnrType1)
all_data <- subset(all_data,select = -MasVnrType1)

#MasVnrArea
all_data$MasVnrArea[which(is.na(all_data$MasVnrArea))] <- 0

#MSZoning
all_data$MSZoning1 <- as.character(all_data$MSZoning)
all_data$MSZoning1[which(is.na(all_data$MSZoning))] <- getmode(all_data$MSZoning)
all_data$MSZoning <- as.factor(all_data$MSZoning1)
all_data <- subset(all_data,select = -MSZoning1)


#Utilities We can safely drop this feature since most of the observations are “AllPub”, 1 is “NoSewa”, and 2 “NA”
unique(all_data$Utilities)

## [1] "AllPub" "NoSeWa" NA
table(all_data$Utilities)
all_data$Utilities = NULL

#Functional
all_data$Functional1 <- as.character(all_data$Functional)
all_data$Functional1[which(is.na(all_data$Functional))] <- "Typ"
all_data$Functional <- as.factor(all_data$Functional1)
all_data <- subset(all_data,select = -Functional1)
#table(all_data$Functional)

#Electrical
all_data$Electrical1 <- as.character(all_data$Electrical)
all_data$Electrical1[which(is.na(all_data$Electrical))] <- getmode(all_data$Electrical)
all_data$Electrical <- as.factor(all_data$Electrical1)
all_data <- subset(all_data,select = -Electrical1)

#KitchenQual
all_data$KitchenQual1 <- as.character(all_data$KitchenQual)
all_data$KitchenQual1[which(is.na(all_data$KitchenQual))] <- getmode(all_data$KitchenQual)
all_data$KitchenQual <- as.factor(all_data$KitchenQual1)
all_data <- subset(all_data,select = -KitchenQual1)

#Exterior1st
all_data$Exterior1st1 <- as.character(all_data$Exterior1st)
all_data$Exterior1st1[which(is.na(all_data$Exterior1st))] <- getmode(all_data$Exterior1st)
all_data$Exterior1st <- as.factor(all_data$Exterior1st1)
all_data <- subset(all_data,select = -Exterior1st1)

#Exterior2nd
all_data$Exterior2nd1 <- as.character(all_data$Exterior2nd)
all_data$Exterior2nd1[which(is.na(all_data$Exterior2nd))] <- getmode(all_data$Exterior2nd)
all_data$Exterior2nd <- as.factor(all_data$Exterior2nd1)
all_data <- subset(all_data,select = -Exterior2nd1)

#SaleType
all_data$SaleType1 <- as.character(all_data$SaleType)
all_data$SaleType1[which(is.na(all_data$SaleType))] <- getmode(all_data$SaleType)
all_data$SaleType <- as.factor(all_data$SaleType1)
all_data <- subset(all_data,select = -SaleType1)
head(all_data,2)


######Converting into factors######
all_data$Street = factor(all_data$Street)
all_data$LotShape = factor(all_data$LotShape)
all_data$LandContour = factor(all_data$LandContour)
all_data$LotConfig = factor(all_data$LotConfig)
all_data$LandSlope = factor(all_data$LandSlope)
all_data$Neighborhood = factor(all_data$Neighborhood)
all_data$Condition1 = factor(all_data$Condition1)
all_data$Condition2 = factor(all_data$Condition2)
all_data$BldgType = factor(all_data$BldgType)
all_data$HouseStyle = factor(all_data$HouseStyle)
all_data$OverallQual = factor(all_data$OverallQual)
all_data$OverallCond = factor(all_data$OverallCond)
all_data$YearBuilt = as.character((all_data$YearBuilt))
all_data$GarageYrBlt = as.character((all_data$GarageYrBlt))
all_data$YearRemodAdd = as.character(all_data$YearRemodAdd)
all_data$RoofStyle = factor(all_data$RoofStyle)
all_data$RoofMatl = factor(all_data$RoofMatl)
all_data$ExterQual = factor(all_data$ExterQual)
all_data$ExterCond = factor(all_data$ExterCond)
all_data$MSSubClass = factor(all_data$MSSubClass)
all_data$Foundation = factor(all_data$Foundation)
all_data$Heating = factor(all_data$Heating)
all_data$HeatingQC = factor(all_data$HeatingQC)
all_data$CentralAir = factor(all_data$CentralAir)
all_data$PavedDrive = factor(all_data$PavedDrive)
all_data$MoSold = factor(all_data$MoSold)
all_data$YrSold = factor(all_data$YrSold)
all_data$SaleCondition = factor(all_data$SaleCondition)
all_data$BsmtFullBath = factor(all_data$BsmtFullBath)
all_data$BsmtHalfBath = factor(all_data$BsmtHalfBath)
all_data$FullBath = factor(all_data$FullBath)
all_data$HalfBath = factor(all_data$HalfBath)

head(all_data, 2)

#######Engineer one feature##########
#linear relationship b/n ground and upper floors
all_data$TotalSF = all_data$TotalBsmtSF + all_data$X1stFlrSF + all_data$X2ndFlrSF
which(colSums(is.na(all_data))>0)


#####normalize the target variable and all skewed variables
all_data_new = all_data
ggplot(all_data, mapping = aes(x = SalePrice)) + geom_histogram(color = "black", fill = "blue")

sapply(all_data_new,class)

#######Grouping data set into numeric and non-numeric#########
Column_classes <- sapply(names(all_data_new),function(x){class(all_data_new[[x]])})
numeric_columns <-names(Column_classes[Column_classes != "factor" & Column_classes != "character"])

#determining skew of each numeric variable
skew <- sapply(numeric_columns,function(x){skewness(all_data_new[[x]],na.rm = T)})
# Let us determine a threshold skewness and transform all variables above the treshold.
skew <- skew[skew > 0.75]
# transform excessively skewed features with log(x + 1)
for(x in names(skew))
{
  all_data_new[[x]] <- log(all_data_new[[x]] + 1)
}
ggplot(all_data_new, mapping = aes(x = SalePrice)) + geom_histogram(color = "black", fill = "blue")


#########Splitting  data(all_data_new) into test/train
train_1 <- all_data_new[all_data_new$isTrain==1,]
test_1 <- all_data_new[all_data_new$isTrain==0,]
dim(train_1)

dim(test_1)

dim(all_data_new)


train_1 = subset(train_1, select = -c(Id, isTrain))
test_1 = subset(test_1, select = -c(Id, isTrain, SalePrice))
set.seed(100)
partition <- createDataPartition(train_1$SalePrice, p = 0.80, list = FALSE)
train.m <- train_1[partition, ]
test.m <- train_1[-partition, ]

head(train.m, 3)

########## MODELING ###########

###FULL MODEL ###
model1 = randomForest(SalePrice ~ ., data = train.m)
summary(model1)

##Plot of model1
plot(model1)

## variable importanace ##
imp = importance(model1)
imp

###Plot of variable importance
varImpPlot(model1)

# RMSE of full model using log of SalePrice
RMSE(test.m$SalePrice, predict(model1, newdata = test.m))

######Prediction with the full model#########
pred = predict(model1, newdata = test.m)
df = data.frame(Pred = exp(pred), Test = exp(test.m$SalePrice))

########## Prediction VS Actual################
head(df, 5)

#########RMSE of the full model using SalePrice ###########
sqrt( mean( (exp(pred)-exp(test.m$SalePrice)) ^2) )


###Reduced Model###
train.rf = train.m
model.select.rf = randomForest(SalePrice ~ TotalSF + MSZoning + LotArea + LotShape + Neighborhood + Condition1 +Condition2+
                                 BldgType +  OverallQual +OverallCond + YearBuilt + RoofMatl + Exterior1st + ExterCond+
                                 Foundation +BsmtQual+ KitchenQual+FullBath+BsmtExposure+Fireplaces+ SaleCondition+
                                 CentralAir+PavedDrive,data = train.rf)  
summary(model.select.rf)

##variable of importance
imp = importance(model.select.rf)
imp

##plot of variable of importance
varImpPlot(model.select.rf)

###model plot
plot(model.select.rf)


######## RMSE of the Reduced model using log of SalePrice ###########
RMSE(test.m$SalePrice, predict(model.select.rf, newdata = test.m))

#####Prediction with the reduced model###
pred = predict(model.select.rf, newdata = test.m)

####Prediction vs Actual####
df = data.frame(Pred = exp(pred), Test = exp(test.m$SalePrice))
head(df, 5)

pred = predict(model.select.rf, newdata = test.m)


####RMSE of the Reduced model using SalePrice####
sqrt( mean( (exp(pred)-exp(test.m$SalePrice)) ^2) )

"Given the RMSE of the full model and the reduce model, we concluded on choosing the reduced model as
our best model. The reduced model returned a similar RMSE to the full model.Thus, the reduced model
matched the full and it will cost less to deploy. It also satisfy the principle of parsimony".
