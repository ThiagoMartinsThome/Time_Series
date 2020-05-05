# State Space Models for Time Series

State space models posit a world in which the true state cannot be measured directly but only inferred from what can be measured. State space models also rely on specifying the dynamics of a system, such as how the true state of the world evolves over time, both due to internal dynamics and the external forces that are applied to a system.

In estimating the underlying state based on observations, we can divide our work into different stages or categories:

  - Filtering
    Using the measurement at time t to update our estimation of the state at time t.

  - Forecasting
    Using the measurement at time t – 1 to generate a prediction for the expected state at time t (allowing us to infer the expected measurement at time t as well).

  - Smoothing
    Using measurement during a range of time that includes t, both before and after it, to estimate what the true state at time t was.

The mechanics of these operations will often be similar, but the distinctions are important. Filtering is a way of deciding how to weigh the most recent information against past information in updating our estimate of state. Forecasting is the prediction of the future state without any information about the future. Smoothing is the use of future and past information in making a best estimate of the state at a given time.

### The Kalman Filter

The Kalman filter is a well-developed and widely deployed method for incorporating new information from a time series and incorporating it in a smart way with previously known information to estimate an underlying state. 

The benefits of the Kalman filter are that it is relatively easy to compute and does not require storage of past data to make present estimates or future forecasts.














