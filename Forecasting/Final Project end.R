############ Final Project ############ 
# Junfei MA | Shanchuan Yu | Rachel Qin
############### FIN 250 ###############
library(fpp2)
library(urca)  
library(normwhn.test)
library(ggplot2)
library(quantmod)
library(rugarch)
library(tidyverse)
library(fGarch)
library(Metrics)
rm(list = ls())

#load data
sp500 <- read.csv("//Users/junfeima/Desktop/FIN 250 Forecasting/Final Project/SP500(days)_2.csv")
str(sp500)
#Department of Homeless Services data
dhs <- read.csv("//Users/junfeima/Desktop/FIN 250 Forecasting/Final Project/DHS_Daily_Report.csv")
str(dhs)

sp500 <- subset(sp500, Date%in% dhs$Date.of.Census)
dhs<- subset(dhs, Date.of.Census %in% sp500$Date)

sp500<-sp500[order(as.Date(sp500$Date, format="%m/%d/%Y")),]
row.names(sp500) <- NULL

dhs<-dhs[order(as.Date(dhs$Date.of.Census, format="%m/%d/%Y")),]
row.names(dhs) <- NULL

#overall plot
##frequency 250
sp500.ts <- ts(sp500$GSPC, start = c(2016,230), frequency = 250) 
dhs.ts <- ts(dhs$Total.Individuals.in.Shelter, start = c(2016,230), frequency = 250) 


plot(sp500.ts.new, xlab = "Time", ylab = "Daily Closing Price",
     main = "S&P500 Daily Closing (11/2016 - 10/2021) ")
plot(dhs.ts, xlab = "Time", ylab = "Daily Individuals",
     main = "DHS Daily Individuals in Shelter (11/2016 - 10/2021) ")

#plot in same plot 
ticks.1 <- seq(2000,4500,500)
ticks.2 <- seq(35000,60000,5000)
plot(sp500.ts, xlab = "Time", ylab = "Daily Closing Price",col="blue",
     main = "S&P500 Daily Closing (11/2016 - 10/2021) ")
axis(2, at=ticks.1, col.ticks="blue", col.axis="blue")
par(new=T)
plot(dhs.ts,yaxt='n', xlab = "", ylab = "",col="red")
axis(4, at=ticks.2, col.ticks="red", col.axis="red")

#ADF Test for stationarity
#sp500 
df.adf <- ur.df(sp500.ts, type = "trend", selectlags = "AIC")
summary(df.adf)
#external 
df.adf_dhs <- ur.df(dhs.ts, type = "trend", selectlags = "AIC")
summary(df.adf_dhs)

#ACF PACF 
acf(sp500.ts)
pacf(sp500.ts)

#Data Partition

nValid <- 350
nTrain <- length(sp500.ts) - nValid
train.ts <- window(sp500.ts,start = c(2016,230), end = c(2016,nTrain+229))
valid.ts <- window(sp500.ts, start = c(2016, nTrain+230), end = c(2016, nTrain+nValid+230))

dhs_train.ts <- window(dhs.ts,start = c(2016,230), end = c(2016,nTrain+229))
dhs_valid.ts <- window(dhs.ts, start = c(2016, nTrain+230), end = c(2016, nTrain+nValid+230))


#Linear Model no season
sp500.exp.fit <- tslm(train.ts ~ trend, lambda = 0)
sp500.exp.pred <- forecast(sp500.exp.fit, h = nValid, level = 0)

#Plot results
plot(train.ts, xlab = "Time",xlim = c(2016.8, 2021.9), ylab = "Daily Closing Price",ylim = c(2000,4500),
     bty = "l", main = "S&P 500 Fit and Forecast with Linear Model")
lines(sp500.exp.pred$mean, col = "red", lwd = 2)
lines(sp500.exp.pred$fitted, lwd = 2, col = "blue")
lines(valid.ts, lwd = 2)

forecast::accuracy(sp500.exp.pred$fitted, train.ts)
forecast::accuracy(sp500.exp.pred$mean, valid.ts)

#Linear Model with season
sp500.exp.fit <- tslm(train.ts ~ trend+season, lambda = 0)
sp500.exp.pred <- forecast(sp500.exp.fit, h = nValid, level = 0)

#Plot results
plot(train.ts, xlab = "Time",xlim = c(2016.8, 2021.9), ylab = "Daily Closing Price",ylim = c(2000,4500),
     bty = "l", main = "S&P 500 Fit and Forecast with Linear Model")
lines(sp500.exp.pred$mean, col = "red", lwd = 2)
lines(sp500.exp.pred$fitted, lwd = 2, col = "blue")
lines(valid.ts, lwd = 2)

forecast::accuracy(sp500.exp.pred$fitted, train.ts)
forecast::accuracy(sp500.exp.pred$mean, valid.ts)


plot(sp500.exp.fit$residuals)
hist(sp500.exp.fit$residuals, ylab = "Frequency", xlab = "Forecast Error", bty = "l", 
     main = "Distribution of residuals from fitted model over training period")
valid.err <- valid.ts - sp500.exp.pred$mean
plot(valid.err)
hist(valid.err, ylab = "Frequency", xlab = "Forecast Error", bty = "l", 
     main = "Distribution of residuals from forecast over validation period")


#ARIMA fit
arima.fit <- auto.arima(train.ts)
summary(arima.fit)
#ARIMA Test
arima.pred <- forecast(arima.fit, h = nValid, level = 0)
#Plot Result
plot(arima.pred, xlab = "Time", ylab = "Passengers", bty = "l",ylim = c(2000, 5000),
     main = "S&P500 (11/2016 - 10/2021) forcast with ARIMA")
lines(arima.pred$fitted, lwd = 2, col = "blue")
lines(sp500.ts)

#ARIMA with external Data
arima.fit_ext <- auto.arima(train.ts, xreg = dhs_train.ts)
summary(arima.fit_ext)
arima.pred <- forecast(arima.fit, h = nValid, level = 0)
#Plot Result
plot(arima.pred, xlab = "Time", ylab = "Passengers", bty = "l",ylim = c(2000, 5000),
     main = "S&P500 (11/2016 - 10/2021) forcast with ARIMA (DHS as External Data)")
lines(arima.pred$fitted, lwd = 2, col = "blue")
lines(sp500.ts)


#Residual
plot(arima.fit$residuals)
hist(arima.fit$residuals, ylab = "Frequency", xlab = "Forecast Error", bty = "l", 
     main = "Distribution of residuals from fitted model over training period")
valid.err <- valid.ts - arima.pred$mean
plot(valid.err)
hist(valid.err, ylab = "Frequency", xlab = "Forecast Error", bty = "l", 
     main = "Distribution of residuals from forecast over validation period")

ggplot(data.frame(residuals = arima.fit$residuals),aes(residuals)) + 
  geom_histogram(bins = 50,aes(y= ..density..),col = "red", fill = "red",alpha = 0.3)+
  geom_density()

#ACF PACF after ARIMA model
acf(arima.fit$residuals)
pacf(arima.fit$residuals)

#accuracy
forecast::accuracy(arima.fit$fitted, train.ts)
forecast::accuracy(arima.pred$mean, valid.ts)


# Garch
diff1sp500<-diff(sp500.ts, lag = 1)
plot(diff1sp500, xlab = "Time", ylab = "difference in index", bty = "l",
     main = "S&P500 (10/2016 - 09/2021) difference")
model3 = garchFit(~arma(5,1) + garch(1,1),diff1train.ts)
plot(model3)
garch.pred <- forecast(model3, h = nValid)
nValid2 <- 400
nTrain2 <- length(diff1sp500) - nValid
train2.ts <- window(diff1sp500, start = c(2016,11), end = c(2016,nTrain2))
valid2.ts <- window(diff1sp500, start = c(2016, nTrain2+1), end = c(2016, nTrain2+nValid2))
g.spec = ugarchspec(mean.model = list(armaOrder = c(5,1)), distribution.model = "std")
g.spec
g.fit = ugarchfit(spec = g.spec, data = train2.ts )
g.fit
g.forecast <- ugarchforecast(g.fit, n.ahead = 10)
plot(g.forecast)
a<-g.forecast@forecast$seriesFor
mae(a,valid2.ts)
rmse(a,valid2.ts)
mape(a,valid2.ts)

#Use Garch model to forecast the future
getSymbols("^GSPC",from="2016-11-1")
head(GSPC)

#   Plot price data for visualization
#   -  "chartSeries" is a function within the quantmod package
chartSeries(GSPC)

#   Could also use "plot" function for data visualization
plot(GSPC$GSPC.Close)

#  Change from price series to returns series
close.GSPC<-GSPC$GSPC.Close
diff1.ts <- diff(close.GSPC, lag = 1)
diff2<-diff1.ts$GSPC.Close
head(diff2,10)
g.spec = ugarchspec(mean.model = list(armaOrder = c(5,1)), distribution.model = "std")
g.spec
g.fit = ugarchfit(spec = g.spec, data =diff2[2:length(diff2)] )
g.fit

#  Using data Within @fit slot, plot the estimated variances and squared residuals
g.var <- g.fit@fit$var
g.res.squared <- (g.fit@fit$residuals)^2

plot(g.res.squared, type = "l", col = "blue")
lines(g.var, col = "red", lwd = 2)

################################################################################

#  Forecast using GARCH model

g.forecast = ugarchforecast(g.fit, n.ahead = 10)
g.forecast
plot(g.forecast,main = "S&P500 lag1 difference forecast for the following ten days ")

#  Results from ugarchforecast are in 2 slots, @model and @forecast
names(g.forecast@model)
names(g.forecast@forecast)

#  Using data within the @ forecast slot, plot the estimated sigma values
g.sig.fore <- g.forecast@forecast$seriesFor
plot(g.sig.fore, type = "l")
