data <- read.csv(file="exam.csv", header=T, row.names=1)
cor <- cor(data)
eigen <- eigen(cor)$values
library(psych)
result <- fa(data, nfactors = length(eigen[eigen > 1]), fm="ml", rotate="varimax", scores="Anderson")
result
result$scores

