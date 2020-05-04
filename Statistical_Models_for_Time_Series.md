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

