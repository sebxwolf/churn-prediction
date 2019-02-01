# Load libraries.
library("doParallel")
library("lattice")
library("ggplot2")
library("caret")
library("randomForest")
library("pROC")
library("dplyr")
library("plyr")


# Set working directory to current folder.
script.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(script.dir)

# Load source files.
source("./io.R")
source("./classDistribution.R")

# Set seed.
set.seed(321)

# Set path to data set.
path <- "./churn.csv"

# Count number of lines in input file.
lines <- readChar(path, file.info(path)$size)
total.rows <- length(gregexpr("\n",lines)[[1L]])
rm(lines)

# Load data.
df <- load.data(p.path = path,
				p.header = TRUE,
				p.dec = ".",
				p.sep = ",",
				p.blank.lines.skip = TRUE,
				p.stringsAsFactors = FALSE,
				p.comment.char = "",
				p.initial.rows = 100,
				p.total.nrows = total.rows,
				p.id = FALSE)

df <- subset(df, select=-c(bcookie,timestamp))

start.time <- as.numeric(as.POSIXct(Sys.time()))

# Remove missing cases.
df <- df[complete.cases(df),]

# Remove redundant features.
# (1) highly correlated
descrCor <- cor(df[2:33])
highCor <- findCorrelation(descrCor, cutoff = .75)
head(highCor)
df <- df[,-highCor]

# (2) linearly dependent
# comboInfo <- findLinearCombos(df[3:30])
# df <- df[, -comboInfo$remove]


# Tranform the class values to factors.
df$cluster[df$cluster == 0] <- "engaged"
df$cluster[df$cluster == 1] <- "churned"
df$cluster <- as.factor(df$cluster)

# Other preprocessing?
df.raw <- df
preProc <- preProcess(df, method = c("range"))
df <- predict(preProc, df)

# Feature engineering.
## Interaction pageviews + dewlltime = intensity of time spent
df <- df %>% mutate(ST_inter_d1 = ST_pageviews_d1 * ST_dwelltime_d1)
df <- df %>% mutate(ST_inter_d2 = ST_pageviews_d2 * ST_dwelltime_d2)
df <- df %>% mutate(ST_inter_d3 = ST_pageviews_d3 * ST_dwelltime_d3)
df <- df %>% mutate(ST_inter_d4 = ST_pageviews_d4 * ST_dwelltime_d4)
df <- df %>% mutate(ST_inter_d5 = ST_pageviews_d5 * ST_dwelltime_d5)
df <- df %>% mutate(ST_inter_d7 = ST_pageviews_d7 * ST_dwelltime_d7)

## Interaction dwelltime + clicks = intensity of time spent (2nd version)
df <- df %>% mutate(ST_inter_d111 = ST_dwelltime_d1 * ST_cliks_d1)
df <- df %>% mutate(ST_inter_d222 = ST_dwelltime_d2 * ST_cliks_d2)
df <- df %>% mutate(ST_inter_d333 = ST_dwelltime_d3 * ST_cliks_d3)
df <- df %>% mutate(ST_inter_d444 = ST_dwelltime_d4 * ST_cliks_d4)
df <- df %>% mutate(ST_inter_d555 = ST_dwelltime_d5 * ST_cliks_d5)
df <- df %>% mutate(ST_inter_d555 = ST_dwelltime_d6 * ST_cliks_d6)
df <- df %>% mutate(ST_inter_d777 = ST_dwelltime_d7 * ST_cliks_d7)

## Interaction pageviews + clicks = intensity of viewing
df <- df %>% mutate(ST_inter_d11 = ST_pageviews_d1 * ST_cliks_d1)
df <- df %>% mutate(ST_inter_d22 = ST_pageviews_d2 * ST_cliks_d2)
df <- df %>% mutate(ST_inter_d33 = ST_pageviews_d3 * ST_cliks_d3)
df <- df %>% mutate(ST_inter_d44 = ST_pageviews_d4 * ST_cliks_d4)
df <- df %>% mutate(ST_inter_d55 = ST_pageviews_d5 * ST_cliks_d5)
df <- df %>% mutate(ST_inter_d77 = ST_pageviews_d7 * ST_cliks_d7)

## Interaction for trends
df <- df %>% mutate(ST_inter_trend = ST_dwelltime_d7 * ST_dwelltime_d1)
df <- df %>% mutate(ST_inter_trend1 = ST_dwelltime_d7 * ST_dwelltime_d6 * ST_dwelltime_d1)

# Set the class.
df <- df %>% select(-cluster,cluster)
class <- length(df)

# Perform stratified bootstrapping (keep 60% of observations for training and 40% for testing).
indices.training <- createDataPartition(df[,class], 
										times = 1, 
										p = .60, 
										list = FALSE)

# Get training and test set.
training <- df[indices.training[,1],]
test  <- df[-indices.training[,1],]

# Print class distribution.
cat("\n\n")
classDistribution(dataset.name = "df",
                  table = df,
                  class = length(df))

classDistribution(dataset.name = "training",
                  table = training,
                  class = length(df))

classDistribution(dataset.name = "test",
                  table = test,
                  class = length(df))

# Specify the tuning parameter grid.
tuneGrid <- expand.grid(.mtry = c(1:10))

# Parallelise
gc()
cores <- makeCluster(detectCores())
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)

# Perform grid search.
control <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        repeats = 1, 
                        search = "grid",
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)

model.training <- train(cluster ~ ., 
                        data = df, 
                        method = "rf", 
                        metric = "ROC", 
                        tuneGrid = tuneGrid, 
                        trControl = control,
                        ntree = 30,
                        verbose = TRUE)

# stop parallelisation
print(paste(c("# Workers: ", getDoParWorkers()), sep = "", collapse = ""))
stopCluster(cl)
gc()

# Print basic information about our model.
print(model.training)
print(model.training$finalModel)

# The plot function can be used to examine the relationship between 
# the estimates of performance and the tuning parameters.
plot(model.training, log = "y")

# Evaluate model on test set.
model.test.pred <- predict(model.training, 
                           test, 
                           type = "raw",
                           norm.votes = TRUE)

model.test.prob <- predict(model.training, 
                           test, 
                           type = "prob",
                           norm.votes = TRUE)

# stop time
end.time <- as.numeric(as.POSIXct(Sys.time()))
tt <- end.time - start.time

# Compute confusion matrix.
performance <- confusionMatrix(model.test.pred, test[, class])
print(performance)
print(performance$byClass)

# Compute AUC for the model.
model.roc <- plot.roc(predictor = model.test.prob[,2],  
                      test[,class],
                      levels = rev(levels(test[,class])),
                      legacy.axes = FALSE,
                      percent = TRUE,
                      mar = c(4.1,4.1,0.2,0.3),
                      identity.col = "red",
                      identity.lwd = 2,
                      smooth = FALSE,
                      ci = TRUE, 
                      print.auc = TRUE,
                      auc.polygon.border=NULL,
                      lwd = 2,
                      cex.lab = 2.0, 
                      cex.axis = 1.6, 
                      font.lab = 2,
                      font.axis = 2,
                      col = "blue")
ciobj <- ci.se(model.roc, specificities = seq(0, 100, 5))
plot(ciobj, type = "shape", col = "#1c61b6AA")
plot(ci(model.roc, of = "thresholds", thresholds = "best"))

print(paste(c("tt: ", tt), sep = "", collapse = ""))
