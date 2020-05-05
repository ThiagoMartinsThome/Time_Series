# Statistical Methods Developed for Time Series

### Advantages and Disadvantages of Statistical Methods for Time Series

**Advantages**

  - These models are simple and transparent, so they can be understood clearly in terms of their parameters.

  - Because of the simple mathematical expressions that define these models, it is possible to derive their properties of interest in a rigorous statistical way.

  - You can apply these models to fairly small data sets and still get good results.

  - These simple models and related modifications perform extremely well, even in comparison to very complicated machine learning models. So you get good performance without the danger of overfitting.

  - Well-developed automated methodologies for choosing orders of your models and estimating their parameters make it simple to generate these forecasts.

**Disadvantages**

  - Because these models are quite simple, they don’t always improve performance when given large data sets. If you are working with extremely large data sets, you may do better with the complex models of machine learning and neural network methodologies.

  - These statistical models put the focus on point estimates of the mean value of a distribution rather than on the distribution. True, you can derive sample variances and the like as some proxy for uncertainty in your forecasts, but your fundamental model offers only limited ways to express uncertainty relative to all the choices you make in selecting a model.

  - By definition, these models are not built to handle nonlinear dynamics and will do a poor job describing data where nonlinear relationships are dominant.

### 1. Linear Regression

A linear regression assumes you have independently and identically distributed (iid) data. unfortunately is not the case with time series data. In time series data, points near in time tend to be strongly correlated with one another. In fact, when there aren’t temporal correlations, time series data is hardly useful for traditional time series tasks, such as predicting the future or understanding temporal dynamics.

But Ordinary least squares linear regression can be applied to time series data provided the following conditions hold:

*Assumptions with respect to the behavior of the time series*

  - The time series has a linear response to its predictors.

  - No input variable is constant over time or perfectly correlated with another input variable. This simply extends the traditional linear regression requirement of independent variables to account for the temporal dimension of the data.

*Assumptions with respect to the error*

  - For each point in time, the expected value of the error, given all explanatory variables for all time periods (forward and backward), is 0.

  - The error at any given time period is uncorrelated with the inputs at any time period in the past or future. So a plot of the autocorrelation function of the errors will not indicate any pattern.

  - Variance of the error is independent of time.
  
 
  If these assumptions hold, then ordinary least squares regression is an unbiased estimator of the coefficients given the inputs, even for time series data. In this case, the sample variances of the estimates have the same mathematical form as they do for standard linear regression.


*Some of the consequences of applying linear regression when your data doesn’t meet the required assumptions are:*

  - Your coefficients will not minimize the error of your model.

  - Your p-values for determining whether your coefficients are nonzero will be incorrect because they rely on assumptions that are not met. This means your assessments of coefficient significance could be wrong.

UNBIASED ESTIMATOR

If an estimate is not an overestimate or underestimate, it is using an unbiased estimator. This tends to be a good thing, although you should be aware of the bias-variance trade-off, which is a description of the problem for both statistical and machine learning problems wherein models with a lower bias in their parameter estimates tend to have a higher variance of the estimate of the parameter. The variance of the parameter’s estimate reflects how variable an estimate will be across different samples of the data.


### Autoregressive Models (AM)

Auto-regressive models (or AR models) assume that a value of a time series depends on the value or on the previous values:

Rt = a1.Rt-1 + a2.Rt-2 + ... + ap.Rt-p + εt

The value of p (that is, the number of lags -or previous values- considered) defines the order of the model. The simplest model is order 1, or AR (1):

AR (1): Rt = a1.Rt-1 + εt

AR (2): Rt = a1.Rt-1 + a2.Rt-2 + εt

AR (p): Rt = a1.Rt-1 + a2.Rt-2 + ... + ap.Rt-p + εt

The AR (1) model is also called the Markov Chain.

εt is white noise: random values independent of the others ε, series with mean and constant variance and null autocorrelation for all lags. 

These values are also often referred to as *shocks*.

      Notice:
      p – The lag value where the PACF chart crosses the upper confidence interval for the first time. 
      q – The lag value where the ACF chart crosses the upper confidence interval for the first time. 

```python

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df, order = (2, 0, 0)).fit()  # model AR of order 2
model.summary()

```
*output*

```python
'''
Statespace Model Results
Dep. Variable:	production	No. Observations:	548
Model:	SARIMAX(2, 0, 0)	Log Likelihood	-1965.427
Date:	Sat, 02 May 2020	AIC	3936.854
Time:	15:56:03	BIC	3949.773
Sample:	01-31-1972	HQIC	3941.904
- 08-31-2017		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ar.L1	1.2575	0.056	22.553	0.000	1.148	1.367
ar.L2	-0.2624	0.057	-4.638	0.000	-0.373	-0.152
sigma2	75.6479	3.187	23.739	0.000	69.402	81.894
Ljung-Box (Q):	1697.19	Jarque-Bera (JB):	125.89
Prob(Q):	0.00	Prob(JB):	0.00
Heteroskedasticity (H):	0.50	Skew:	0.28
Prob(H) (two-sided):	0.00	Kurtosis:	5.28


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
```
AIC and BIC allow us to estimate the correctness of the model:

AIC (Akaike Information Criterion) is a metric that tells us how good or bad our model is. The better predictions a model makes, the less AIC it will have. Although frequently a higher order in the model can lead to better predictions, it is normal that it is at the cost of overtraining it, so AIC imposes a penalty on high-order models (those with many parameters).

BIC (Bayesian Information Criterion) is a similar metric to AIC: the better the model, the lower BIC you will get. BIC also penalizes more complex models, more than AIC, so BIC will tend to suggest the use of simpler models than AIC. In most cases the suggested model will be the same but, when it is not, we must choose:

  - If we look for the best predictive model, AIC will be more advisable.
  - If, on the other hand, we look for a model that explains our data more easily, we will opt for BIC.

```python
IN:
prediction = model.get_forecast(steps = 12) # get prediction
model.predict(start = len(df), end = len(df) + 12)

'''
OUT:
2017-09    116.518645
2017-10    116.597866
2017-11    116.052706
2017-12    115.346361
2018-01    114.601152
2018-02    113.849361
2018-03    113.099493
2018-04    112.353769
2018-05    111.612753
2018-06    110.876568
2018-07    110.145224
2018-08    109.418700
2018-09    108.696967
Freq: M, dtype: float64
'''

IN:
# Access the predicted values of the data in which the model has been trained
model.fittedvalues

'''
OUT:
date
1972-01      0.000000
1972-02     85.365009
1972-03     67.830866
1972-04     64.181334
1972-05     63.868445
              ...    
2017-04    102.551244
2017-05    107.485745
2017-06     99.980975
2017-07    104.295011
2017-08    101.664116
Freq: M, Length: 548, dtype: float64
'''

# Plot the values

fig, ax = plt.subplots()
df.truncate(before = "2010").plot(ax = ax)
model.fittedvalues.truncate(before = "2010").plot(ax = ax)
plt.show();

# Calculate the mean square error
from sklearn.metrics import mean_squared_error
mean_squared_error(df.production, model.fittedvalues)
```

### Moving Average Models (MA)

Moving-average models consider that a value of a time series depends on the ε values of the previous values:

Rt = m1.εt-1 + m2.εt-2 + ... + mq.εt-q + εt

Again, the value of q (number of lags or previous values to use) defines the order of the model:

MA (1): Rt = m1.εt-1 + εt

MA (2): Rt = m1.εt-1 + m2.εt-2 + εt

MA (q): Rt = m1.εt-1 + m2.εt-2 + ... + mq.εt-q + εt

      Notice:
      p – The lag value where the PACF chart crosses the upper confidence interval for the first time. 
      q – The lag value where the ACF chart crosses the upper confidence interval for the first time. 

```phyton
model = SARIMAX(df, order = (0, 0, 3)).fit()  

'''
order - The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters. d must be an integer indicating the integration order of the process, while p and q may either be an integers indicating the AR and MA orders (so that all lags up to those orders are included) or else iterables giving specific AR and / or MA lags to include. Default is an AR(1) model: (1,0,0).
'''
model.summary()

'''
Statespace Model Results
Dep. Variable:	production	No. Observations:	548
Model:	SARIMAX(0, 0, 3)	Log Likelihood	-2510.939
Date:	Sat, 02 May 2020	AIC	5029.877
Time:	15:56:05	BIC	5047.102
Sample:	01-31-1972	HQIC	5036.610
- 08-31-2017		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ma.L1	1.9476	0.046	42.340	0.000	1.857	2.038
ma.L2	1.7465	0.068	25.843	0.000	1.614	1.879
ma.L3	0.6080	0.046	13.081	0.000	0.517	0.699
sigma2	553.5424	47.782	11.585	0.000	459.892	647.193
Ljung-Box (Q):	2271.89	Jarque-Bera (JB):	2.17
Prob(Q):	0.00	Prob(JB):	0.34
Heteroskedasticity (H):	1.11	Skew:	0.14
Prob(H) (two-sided):	0.47	Kurtosis:	3.15


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
# Prediction with confiance interval
prediction = model.get_forecast(steps = 12)
lower = prediction.conf_int()["lower production"]
upper = prediction.conf_int()["upper production"]

#plot
fig, ax = plt.subplots()
df.truncate(before = "2005").plot(ax = ax)
prediction.predicted_mean.plot(ax = ax)
ax.fill_between(lower.index, lower, upper, alpha = 0.4)
plt.show()

```

### Autoregressive Moving Average Models

The ARMA model (auto-regressive moving average model) is a combination of the AR and MA models:

ARMA = AR + MA

The values of the time series are estimated from the previous values of the series (AR model) and from the previous epsilon (MA model). The order of an ARMA model is made up of the order of the AR and MA models that comprise it:

ARMA (1, 1): Rt = a1.Rt-1 + m1.εt-1 + εt

ARMA (p, q): AR (p): Rt = a1.Rt-1 + a2.Rt-2 + ap.Rt-p + Rt + m1.εt-1 + m2.εt-2 + ... + mq.εt-q + εt

      Notice:
      p – The lag value where the PACF chart crosses the upper confidence interval for the first time. 
      q – The lag value where the ACF chart crosses the upper confidence interval for the first time. 

```python
model = SARIMAX(df, order = (2, 0, 3)).fit()
model.summary()
'''
Statespace Model Results
Dep. Variable:	production	No. Observations:	548
Model:	SARIMAX(2, 0, 3)	Log Likelihood	-1915.281
Date:	Sat, 02 May 2020	AIC	3842.562
Time:	15:56:06	BIC	3868.400
Sample:	01-31-1972	HQIC	3852.661
- 08-31-2017		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ar.L1	0.0008	0.001	0.542	0.588	-0.002	0.004
ar.L2	0.9992	0.001	674.963	0.000	0.996	1.002
ma.L1	1.3974	0.047	29.963	0.000	1.306	1.489
ma.L2	0.5834	0.071	8.259	0.000	0.445	0.722
ma.L3	0.1765	0.045	3.898	0.000	0.088	0.265
sigma2	61.9489	3.463	17.887	0.000	55.161	68.737
Ljung-Box (Q):	1195.41	Jarque-Bera (JB):	56.43
Prob(Q):	0.00	Prob(JB):	0.00
Heteroskedasticity (H):	0.53	Skew:	0.32
Prob(H) (two-sided):	0.00	Kurtosis:	4.44


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
prediction = model.get_forecast(steps = 12)
lower = prediction.conf_int()["lower production"]
upper = prediction.conf_int()["upper production"]

fig, ax = plt.subplots()
candy.truncate(before = "2010").plot(ax = ax)
prediction.predicted_mean.plot(ax = ax)
ax.fill_between(lower.index, lower, upper, alpha = 0.4)
plt.show()
```

ARMAX

An extension of the ARMA model is one that includes external (exogenous) data to create the model. For example, we can have a time series in which, for each day of the year, the sales made are indicated. But we could also include information about what day of the week each day is, or what days are holidays.

The result is a kind of combination between an ARMA model and a linear regression.

The equations that define an ARMA model and another ARMAX model (in both cases of order (1, 1)) are the following:

ARMA (1,1): yt = a1yt-1 + m1εt-1 + εt

ARMAX (1,1): yt = a1yt-1 + m1εt-1 + εt + x1zt

The ARMAX model is created with the same ARMA class that we have seen adding the exog parameter.

For example, let's calculate the evolution of the following sales adding as exogenous information if it was a holiday or not:

```python
sales = pd.Series([10, 22, 15, 10, 25, 30, 18])
sales.plot();

holidays = pd.Series([1, 0, 0, 0, 0, 1, 1])
model = SARIMAX(sales, order = (2, 0, 1), exog = holidays).fit()
exog_data = np.array([0, 0]).reshape(-1, 1)
prediction = model.get_forecast(steps = 2, exog = exog_data)
prediction.predicted_mean
prediction.conf_int()
```

### Autoregressive Integrated Moving Average Models

The difference between an ARMA model and an ARIMA model is that the ARIMA model includes the term integrated, which refers to how many times the modeled time series must be differenced to produce stationarity.

Differencing is converting a time series of values into a time series of changes in values over time. Most often this is done by calculating pairwise differences of adjacent points in time, so that the value of the differenced series at a time t is the value at time t minus the value at time t – 1. However, differencing can also be performed on different lag windows, as convenient.
The ARIMA model is specified in terms of the parameters (p, d, q). We select the values of p, d, and q that are appropriate given the data we have. "d" is the number of times the series is to be differentiated.

Here are some well-known examples from the Wikipedia description of ARIMA models:

  - ARIMA(0, 0, 0) is a white noise model.

  - ARIMA(0, 1, 0) is a random walk, and ARIMA(0, 1, 0) with a nonzero constant is a random walk with drift.

  - ARIMA(0, 1, 1) is an exponential smoothing model, and an ARIMA(0, 2, 2) is the same as Holt’s linear method, which extends exponential smoothing to data with a trend, so that it can be used to forecast data with an underlying trend.

The order parameter is a tuple (p, d, q) in which p is the order of the AR model, q the order of the MA model and d the number of times the series is to be differentiated.

```python

model = SARIMAX(candy, order = (3, 1, 2)).fit()
prediction = model.get_forecast(steps = 24)
lower = prediction.conf_int()["lower production"]
upper = prediction.conf_int()["upper production"]
prediction.predicted_mean.head()

```

### Seasonal Autoregressive Integrated Moving Average Models

Now that we are able to extract seasonality from a time series, we can use it to improve our predictions. For this we will make use of the SARIMA or "Seasonal ARIMA" model.

Training a SARIMA model is like training two ARIMA models: one for the seasonal part and the other for the rest of the information. That is why we will have not one, but two orders: (p, d, q) for the ARIMA model with non-seasonal components, and (P, D, Q) S for the ARIMA model with seasonal components, with S being periodicity value (that is, 7 arguments are required).

      S is an integer giving the periodicity (number of periods in season), often it is 4 for quarterly data or 12 for monthly data. Default is no seasonal effect.

Comparing the expressions of the ARIMA and SARIMA models:

  - ARIMA (2, 0, 1): yt = a1yt-1 + a2yt-2 + m1εt-1 + εt

  - SARIMA (0, 0, 0) (2, 0, 1) 7: yt = a7yt-7 + a14yt-14 + m7εt-7 + εt

The previous ARIMA model will be able to capture patterns from one period to the next, but will not be able to capture information regarding periodicity. In contrast, the SARIMA model shown may capture seasonal patterns, but not patterns that follow data from one lag to the next.

Adding both approaches, we will be able to capture all the existing patterns.

The SARIMAX class supports this functionality, being the way of use is very similar to what has already been seen:

model = SARIMAX (dataframe, order = (p, i, q), seasonal_order = (P, I, Q, S), trend = "c")

The trend parameter is used to indicate to the algorithm that our data is not centered around the zero value.

      TREND - Parameter controlling the deterministic trend polynomial 𝐴(𝑡). Can be specified as a string where ‘c’ indicates a constant (i.e. a degree zero component of the trend polynomial), ‘t’ indicates a linear trend with time, and ‘ct’ is both. Can also be specified as an iterable defining the polynomial as in numpy.poly1d, where [1,1,0,1] would denote 𝑎+𝑏𝑡+𝑐𝑡3. Default is to not include a trend component.

```python
model = SARIMAX(df.to_timestamp(), order = (3, 1, 2), seasonal_order = (1, 1, 2, 12)).fit()

# Convert the index of the dataframe to timestamp to avoid problems with the plot_diagnostics function
#Again, we must ensure that the seasonal component is stationary, 
#for which we must also resort to the appropriate transformations.

prediction = model.get_forecast(steps = 12)
lower = prediction.conf_int()["lower production"]
upper = prediction.conf_int()["upper production"]
df.truncate(before = "2010").plot()
prediction.predicted_mean.plot()
plt.fill_between(lower.index, lower, upper, alpha = 0.4)
plt.show()

```
ARIMA models are trained using the Maximum Likelihood procedure, which looks for those parameters that maximize the probability that the training data is what it really is.

Some of this information is found in the training summary:

```python
model.summary()

'''
Statespace Model Results
Dep. Variable:	production	No. Observations:	548
Model:	SARIMAX(3, 1, 2)x(1, 1, 2, 12)	Log Likelihood	-1466.780
Date:	Sat, 02 May 2020	AIC	2951.561
Time:	15:56:54	BIC	2990.101
Sample:	01-01-1972	HQIC	2966.640
- 08-01-2017		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ar.L1	0.2114	0.281	0.753	0.452	-0.339	0.762
ar.L2	0.3559	0.197	1.810	0.070	-0.030	0.741
ar.L3	0.1679	0.049	3.460	0.001	0.073	0.263
ma.L1	-0.5107	0.284	-1.801	0.072	-1.067	0.045
ma.L2	-0.4389	0.265	-1.654	0.098	-0.959	0.081
ar.S.L12	-0.0014	0.425	-0.003	0.997	-0.834	0.831
ma.S.L12	-0.6382	0.421	-1.515	0.130	-1.464	0.187
ma.S.L24	-0.1083	0.302	-0.359	0.720	-0.700	0.483
sigma2	13.8135	0.710	19.467	0.000	12.423	15.204
Ljung-Box (Q):	80.35	Jarque-Bera (JB):	26.52
Prob(Q):	0.00	Prob(JB):	0.00
Heteroskedasticity (H):	1.32	Skew:	-0.25
Prob(H) (two-sided):	0.06	Kurtosis:	3.97


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
```
The Ljung-Box test starts from the null hypothesis that the residuals have a null correlation for all lags. Prob (Q) is the corresponding p-value.

The Jarque-Bera (JB) test starts from the null hypothesis that the residuals have a Gaussian distribution, and Prob (JB) is their p-value.

AUTOMATION OF OPTIMAL ORDER CALCULATION

Although we can still use loops to calculate the optimal orders, now that we have 7 orders to define the complete model, this method is impractical. Fortunately the pmdarima library offers us a function that does this job for us, the auto_arima function.

      http://alkaline-ml.com/pmdarima/1.0.0/modules/generated/pmdarima.arima.auto_arima.html

```python
# install and import the library
!pip install pmdarima
import pmdarima as pm

model = pm.auto_arima(
    df,
    m = 12,
    suppress_warnings=True
)
```
This function returns an object of the ARIMA class, implementation of the pmdarima algorithm.



```python
model.summary()
'''
Statespace Model Results
Dep. Variable:	y	No. Observations:	548
Model:	SARIMAX(1, 1, 2)x(2, 0, 1, 12)	Log Likelihood	-1511.345
Date:	Sat, 02 May 2020	AIC	3038.689
Time:	18:11:06	BIC	3073.125
Sample:	0	HQIC	3052.150
- 548		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
intercept	4.07e-05	0.000	0.189	0.850	-0.000	0.000
ar.L1	0.8555	0.039	21.680	0.000	0.778	0.933
ma.L1	-1.1563	0.057	-20.302	0.000	-1.268	-1.045
ma.L2	0.1766	0.051	3.465	0.001	0.077	0.277
ar.S.L12	1.1673	0.058	20.153	0.000	1.054	1.281
ar.S.L24	-0.1721	0.057	-3.017	0.003	-0.284	-0.060
ma.S.L12	-0.7900	0.039	-20.120	0.000	-0.867	-0.713
sigma2	13.9257	0.704	19.792	0.000	12.547	15.305
Ljung-Box (Q):	96.75	Jarque-Bera (JB):	33.62
Prob(Q):	0.00	Prob(JB):	0.00
Heteroskedasticity (H):	1.18	Skew:	-0.18
Prob(H) (two-sided):	0.26	Kurtosis:	4.16


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
```

```python
fig = model.plot_diagnostics()
fig.set_size_inches(15, 9)
plt.show();

forecast_mean, conf_int = model.predict(24, return_conf_int = True)

forecast_mean
conf_int[:5]

forecast_lower = conf_int[:, 0]
forecast_upper = conf_int[:, 1]

dates = pd.period_range(start = df.index[-1], periods = 25, freq = "M")[1:]

forecast = pd.Series(forecast_mean, index = dates)

fig, ax = plt.subplots()
df.truncate(before = "2010").plot(ax = ax)
forecast.plot(ax = ax, color = "red")
plt.fill_between(dates, forecast_lower, forecast_upper, color = "lightblue", alpha = 0.6)
plt.show()


```



```python

```
