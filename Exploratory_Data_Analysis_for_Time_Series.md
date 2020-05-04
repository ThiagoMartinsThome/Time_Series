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
