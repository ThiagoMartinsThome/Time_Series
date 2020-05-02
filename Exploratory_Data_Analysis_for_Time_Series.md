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
```
out:
(-1.8870498112252774,
 0.3381775973004306,
 14,
 533,
 {'1%': -3.442678467240966,
  '10%': -2.5696661916864083,
  '5%': -2.8669778698997543},
 3012.7890909742596)

This test seeks to determine the existence of unit roots in the series or not (the null hypothesis of this test is that there is a unit root, that is, that the series is not stationary).

The first element is the test statistic: the more negative, the more likely the series will be stationary.

The second element is the p-value: the probability of the statistic under study. **If it is less than 0.05**, we can reject the null hypothesis and assume that the time series is stationary.

The last values are the critical values: the p-values for different confidence intervals.

In any case, it is always convenient to perform a visual analysis of the data.

It is often the case that a time series can be made stationary enough with a few simple transformations. A log transformation and a square root transformation are two popular options, particularly in the case of changing variance over time. Likewise, removing a trend is most commonly done by differencing. Sometimes a series must be differenced more than once. However, if you find yourself differencing too much (more than two or three times) it is unlikely that you can fix your stationarity problem with differencing.

Another common but distinct assumption is the normality of the distribution of the input variables or predicted variable. In such cases, other transformations may be necessary. A common one is the Box Cox transformation, which is implemented in scipy.stats in Python. The transformation makes non-normally distributed data (skewed data) more normal. However, just because you can transform your data doesn’t mean you should. Think carefully about the meaning of the distances between data points in your original data set before transforming them, and make sure, regardless of your task, that transformations preserve the most important information.
