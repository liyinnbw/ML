# An Attempt to Predict Future Asset Price Change
A runnable version of the code is available at [this Google Colab notebook](https://colab.research.google.com/drive/1PYPMb_-QsIgdmG4jEG-9ZuuxJMKuY4S2)

## Problem Definition
Given a time series of past N days of an asset's trading data (including price, volume, etc.) predict the average price change in next M days.

## Input
N days of daily close price & volume changes.

## Output
Average close price change over the next M days.

## Model Used
LSTM.

## Conclusion
The model FAILED to predict correctly.

I have tried other ways, including:
* Longer or shorter input history period
* Longer or shorter prediction day range
* Different models (MLP, 1D-Convolution, LSTM)
* Different features (SMA, MACD, RSI...)

All does not work, which makes me conclude that either I did something wrong or asset price does "random walks". If you get better results, please let me know.
