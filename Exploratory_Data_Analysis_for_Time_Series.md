# Exploratory Data Analysis for Time Series

STATIONARITY

A stationary time series is one that has fairly stable statistical properties over time, particularly with respect to mean and variance.

So we **don't** expect that:
  - The mean value is increasing over time.
  - The variance of the process is increasing over time.
  - The process displays strong seasonal behavior.

Unfortunately the most of the datasets available have non-stationary data. To apply predictive models we need to transform the data into stationary series.

Tests for determining whether a process is stationary are called hypothesis tests. The Augmented Dickey–Fuller (ADF) test is the most commonly used metric to assess a time series for stationarity problems. 

Example:
```python
from statsmodels.tsa.stattools import adfuller

adfuler()
'''
out:
(-1.8870498112252774,
 0.3381775973004306,
 14,
 533,
 {'1%': -3.442678467240966,
  '10%': -2.5696661916864083,
  '5%': -2.8669778698997543},
 3012.7890909742596)
 '''
 
# Function to test the Stationary and print a Fuller table
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

```


This test seeks to determine the existence of unit roots in the series or not (the null hypothesis of this test is that there is a unit root, that is, that the series is not stationary).

The first element is the test statistic: the more negative, the more likely the series will be stationary.

The second element is the p-value: the probability of the statistic under study. **If it is less than 0.05**, we can reject the null hypothesis and assume that the time series is stationary.

The last values are the critical values: the p-values for different confidence intervals.

In any case, it is always convenient to perform a visual analysis of the data.

It is often the case that a time series can be made stationary enough with a few simple transformations. A log transformation and a square root transformation are two popular options, particularly in the case of changing variance over time. Likewise, removing a trend is most commonly done by differencing. Sometimes a series must be differenced more than once. However, if you find yourself differencing too much (more than two or three times) it is unlikely that you can fix your stationarity problem with differencing.

Another common but distinct assumption is the normality of the distribution of the input variables or predicted variable. In such cases, other transformations may be necessary. A common one is the Box Cox transformation, which is implemented in scipy.stats in Python. The transformation makes non-normally distributed data (skewed data) more normal. However, just because you can transform your data doesn’t mean you should. Think carefully about the meaning of the distances between data points in your original data set before transforming them, and make sure, regardless of your task, that transformations preserve the most important information.

ROLLING WINDOWS (Moving averages)

Is a time series function where you can aggregate data either to compress it (downsampling), to smooth it or for informative exploratory visualizations.

In a very simple words we take a window size of k at a time and perform some desired mathematical operation on it. A window of size k means k consecutive values at a time. In a very simple case all the ‘k’ values are equally weighted.

```python
df.rolling(3).mean() 
```

EXPANDING WINDOWS

Expanding windows are less commonly used in time series analysis than rolling windows because their proper application is more limited. Expanding windows make sense only in cases where you are estimating a summary statistic that you believe is a stable process rather than evolving over time or oscillating significantly. An expanding window starts with a given minimum size, but as you progress into the time series, it grows to include every point up to a given time rather than only a finite and constant size.


```python
df.expanding(2).sum()
```
THE AUTOCORRELATION FUNCTION (ACF)

Autocorrelation, also known as serial correlation, is the correlation of a signal with a delayed copy of itself as a function of the delay. Informally, it is the similarity between observations as a function of the time lag between them.

```python
from statsmodels.tsa.stattools import acf

acf(y, nlags = 30, unbiased = True, fft = False)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(y, lags = 20, unbiased = True);
```
From the visual analysis of the autocorrelation function we can determine the periodicity of our time series.

THE PARTIAL AUTOCORRELATION FUNCTION (PACF)

It is defined as the set of results of calculating the autocorrelation of the series with itself displaced by a number of lags, eliminating the influence of minor lags.

The partial autocorrelation function reveals which correlations are “true” informative correlations for specific lags rather than redundancies.

```python
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

pacf(x, nlags = 29, method = "ywm")
plot_pacf(x, lags = 40, method = "ywm");
```
WHITE NOISE

Time series that has mean value, constant variance and null autocorrelation for all lags.


TRANSFORMATIONS

Estimating & Eliminating Trend

One of the first tricks to reduce trend can be transformation. For that we can apply transformation which penalize higher values more than smaller values. These can be taking a log, square root, cube root, etc. 

  - Log transformation:
```pyhton
ts_log = np.log(ts)
plt.plot(ts_log)
```
Ususaly with this transformation is easy to see a forward trend in the data. But its not very intuitive in presence of noise. So we can use some techniques to estimate or model this trend and then remove it from the series. There can be many ways of doing it and some of most commonly used are:

    - Aggregation – taking average for a time period like monthly/weekly averages
    - Smoothing – taking rolling averages
    - Polynomial Fitting – fit a regression model

  - Moving Average:
  
```python
moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

#Difference
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff) #Code example in fuller topic
```

However, a drawback in this particular approach is that the time-period has to be strictly defined. In this case we can take yearly averages but in complex situations like forecasting a stock price, its difficult to come up with a number. So we take a ‘weighted moving average’ where more recent values are given a higher weight. There can be many technique for assigning weights. A popular one is exponentially weighted moving average where weights are assigned to all the previous values with a decay factor.

```python
expwighted_avg = pd.ewm(ts_log, halflife=12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)
````
  - Differencing:
  
Taking the differece with a particular time lag.

```python
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
````

  - Percentaje change:
  
```python
ts_pct = ts.pct_change().dropna()
plt.plot(ts_pct)
```

Other transformations can be:

    - Calculate the square root of the data: np.sqrt (ts)
    - Consider proportional change: ts.shift (1) / ts
    - The call log-return: np.log (ts / ts.shift (1))
  
  
  - Decomposition:

Modeling both trend and seasonality and removing them from the model.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)
#decomposition = seasonal_decompose(x = candy.production.to_timestamp())
#Is necesary that the index must be converted to timestamp

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
```
TAKING IT BACK TO ORIGINAL SCALE

Since the combined model gave best result, lets scale it back to the original values and see how well it performs there. First step would be to store the predicted results as a separate series and observe it.



```python
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()
```
Because we took a lag by 1 and first element doesn’t have anything before it to subtract from. The way to convert the differencing to log scale is to add these differences consecutively to the base number. An easy way to do it is to first determine the cumulative sum at index and then add it to the base number. The cumulative sum can be found as:

```python
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print predictions_ARIMA_diff_cumsum.head()
```
You can quickly do some back of mind calculations using previous output to check if these are correct. Next we’ve to add them to base number. For this lets create a series with all values as base number and add the differences to it. This can be done as:

```python
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
```
Here the first element is base number itself and from thereon the values cumulatively added. Last step is to take the exponent and compare with the original series.

```python
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
```

DIAGNOSIS OF THE MODEL

Once the model is obtained, we can evaluate to what extent it adequately adapts to our data by analyzing the residuals, to which we have access through the .resid attribute:

      model.resid.head()
      
In an ideal model, the residuals should be zero-centered Gaussian white noise, that is, values with zero correlation at all lags.

The .plot_diagnostics () method shows four graphs that allow us to make this evaluation

```python
model.plot_diagnostics(figsize = (14, 8))
plt.show()
```
Reading the graphs:

The upper left figure shows the standardized one-step-ahead residuals. In a suitable model, no structure should be shown.

The upper right figure shows the distribution of the residuals. The red line shows a normal distribution and the green line shows the KDE (Kernel Density Estimation) of our data. If the model is correct, both curves should be very close.

The lower left image shows the Q-Q graph, which shows the probability distribution of a population from which a random sample has been drawn and a distribution that is used for comparison. If the residuals follow a normal distribution, all points should be on the red line, except, perhaps, some points at the ends.

The last graph, at the bottom right, shows a correlogram, which is nothing more than a graph of the ACF function of the residuals rather than the data. Except for the value corresponding to zero lag, all the others should be zero (that is, be within the 5% range indicated by the blue line). If the autocorrelation is not null for any lag, it is because there is information in the data that our model has not captured.
