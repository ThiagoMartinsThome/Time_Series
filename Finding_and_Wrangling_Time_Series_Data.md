# Finding and Wrangling Time Series Data

## 1. Definitions 

UNIVARIATE AND MULTIVARIATE TIME SERIES

Univariate time series: Just one variable measured against time.

Multivariate time series: With multiple variables measured at each timestamp. They are particularly rich for analysis because often the measured variables are interrelated and show temporal dependencies between one another. 

LOOKAHEAD

The term lookahead is used in time series analysis to denote any knowledge of the future. You shouldn’t have such knowledge when designing, training, or evaluating a model. A lookahead is a way, through data, to find out something about the future earlier than you ought to know it.

A lookahead is any way that information about what will happen in the future might propagate back in time in your modeling and affect how your model behaves earlier in time. For example, when choosing hyperparameters for a model, you might test the model at various times in your data set, then choose the best model and start at the beginning of your data to test this model. This is problematic because you chose the model for one time knowing things that would happen at a subsequent time—a lookahead.

Unfortunately, there is no automated code or statistical test for a lookahead, so it is something you must be vigilant and thoughtful about.

## 2. Cleaning Data

### Handling Missing Data

The most common methods to address missing data in time series are:

IMPUTATION

When we fill in missing data based on observations about the entire data set.

  - FORWARD FILL : Is to carry forward the last known value prior to the missing one. 
  
  - BACKWARD FILL: Is to carry backward the last known value prior to the missing one. However, this is a case of a lookahead, so you should only do this when you are not looking to predict the future from the data and when, from domain knowledge, it makes more sense to fill in data backward rather than forward in time.
  
```python
df.ffill() # forward fill

df.bfill() # backward fill
```
  
  - MOVING AVERAGE: We can also impute data with either a rolling mean or median. Known as a moving average, this is similar to a forward fill in that you are using past values to “predict” missing future values (imputation can be a form of prediction). With a moving average, however, you are using input from multiple recent times in the past.

  A moving average doesn’t have to be an arithmetic average. For example, exponentially weighted moving averages would give   more weight to recent data than to past data. Alternately, a geometric mean can be helpful for time series that exhibit     strong serial correlation and in cases where values compound over time.
  
  A rolling mean data imputation reduces variance in the data set. This is something you need to keep in mind when        calculating accuracy, R² statistics, or other error metrics. Your calculation may overestimate your model’s performance, a    frequent problem when building time series models.

```python
# SMA(simple moving average) code example
rolling_mean = df.y.rolling(window=20).mean() # 20 days
rolling_mean2 = df.y.rolling(window=50).mean() # 50 days

# EMA(exponential moving average) code example
exp1 = df.y.ewm(span=20, adjust=False).mean() # 20 days
exp2 = df.y.ewm(span=50, adjust=False).mean() # 50 days

# GM (geometrical mean) code example
from scipy.stats.mstats import gmean

x = np.random.uniform(0,1,260)  
geo_mean = pd.Series(x).rolling(window=30).apply(gmean) # 30 days
```
  
INTERPOLATION

Is a method of determining the values of missing data points based on geometric constraints regarding how we want the overall data to behave. For example, a linear interpolation constrains the missing data to a linear fit consistent with known neighboring points.

Linear interpolation is particularly useful and interesting because it allows you to use your knowledge of how your system behaves over time.

```python
# Linear interpolation code example
df.interpolate(method='linear', limit_direction='forward', axis=0) 
```

DELETION OF AFFECTED TIME PERIODS

When we choose not to use time periods that have missing data at all.

```python
# Fill missing values code example
df.fillna()

# Delete missing values code example
df.dropna(inplace=True)
```
UPSAMPLING AND DOWNSAMPLING

Downsampling is subsetting data such that the timestamps occur at a lower frequency than in the original time series. Upsampling is representing data as if it were collected more frequently than was actually the case.

```python
# Downsampling code example
df.resample('5Min')

# Upsampling and fill NaN values method forward filling code example
df.resample('30S').ffill()
```
SMOOTHING DATA

Smoothing can serve a number of purposes:

  - Data preparation : Is your raw data unsuitable? For example, you may know very high values are unlikely or unphysical, but you need a principled way to deal with them. Smoothing is the most straightforward solution.

  - Feature generation: The practice of taking a sample of data, be it many characteristics about a person, image, or anything else, and summarizing it with a few metrics. In this way a fuller sample is collapsed along a few dimensions or down to a few traits. Feature generation is especially important for machine learning.

  - Prediction: The simplest form of prediction for some kinds of processes is mean reversion, which you get by making predictions from a smoothed feature.

  - Visualization: To reduce the noise of the graph.
  
  **Hyperparameters**: Simply the numbers you use to tune, or adjust, the performance of a statistical or machine learning model. To optimize a model, you try several variations of the hyperparameters of that model, often performing a grid search to identify these parameters. A grid search is exactly that: you construct a “grid” of all possible combinations of hyperparameters in a search space and try them all.
  
EXPONENTIAL SMOOTHING

For a given window, the nearest point in time is weighted most heavily and each point earlier in time is weighted exponentially less. 

```python
# Exponential smoothing average code example
exp1 = df.y.ewm(span=20, alpha=.5, adjust=True).mean() # 20 days
```
Note that a simple exponential smoothing does not perform well (for prediction) in the case of data with a long-term trend. Holt’s Method and Holt–Winters smoothing , are two exponential smoothing methods applied to data with a trend, or with a trend and seasonality.

There are many other widely used smoothing techniques. For example, Kalman filters smooth the data by modeling a time series process as a combination of known dynamics and measurement error. LOESS (short for “locally estimated scatter plot smoothing”) is a nonparametric method of locally smoothing data. These methods, and others, gradually offer more complex ways of understanding smoothing but at increased computational cost. Of note, Kalman and LOESS incorporate data both earlier and later in time, so if you use these methods keep in mind the leak of information backward in time, as well as the fact that they are usually not appropriate for preparing data to be used in forecasting applications. *Statsmodels provides models for smoothing methods. Ex. from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt*

## 3. Seasonal Data


SEASONAL DATA AND CYCLICAL DATA

**Seasonal** time series are time series in which behaviors recur over a fixed period. There can be multiple periodicities reflecting different tempos of seasonality, such as the seasonality of the 24-hour day versus the 12-month calendar season, both of which exhibit strong features in most time series relating to human behavior.

**Cyclical** time series also exhibit recurring behaviors, but they have a variable period. A common example is a business cycle, such as the stock market’s boom and bust cycles, which have an uncertain duration. Likewise, volcanoes show cyclic but not seasonal behaviors. We know the approximate periods of eruption, but these are not precise and vary over time.

## 4. Time Zones

Time zones are intrinsically tedious, painful, and difficult to get right even with a lot of effort. There are many reasons for this:

  - Time zones are shaped by political and social decisions.

  - There is no standard way to transport time zone information between languages or via an HTTP protocol.
  
  - There is no single protocol for naming time zones or for determining start and end dates of daylight savings offsets.

  - Because of daylight savings, some times occur twice a year in their time zones.
  
  The main libraries you are likely to use are datetime, pytz, and dateutil. Notice that if we are comparing an object with a time zone to one without a time zone or calculating time deltas will cause a TypeError.
