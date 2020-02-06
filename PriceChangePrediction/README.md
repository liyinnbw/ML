# An Attempt to Predict Future Asset Price Change
A runnable version of the code is available at [this Google Colab notebook](https://drive.google.com/open?id=1MGt3h0nlBYjaxgZeK7z9OjEYaY-T-p0D).

## Problem Definition
Given a time series of past N days of an asset's trading data (including price, volume, etc.) predict the average price change in next M days. Towards the end of the notebook, I also use the trained model to do trading.

## Input
N days of daily close price & volume changes.

## Output
Average close price change over the next M days.

## Model Used
LSTM.

## Conclusion
* LSTM is able to predict price change when using some combinations of common technical indicators as input feature.
* LSTM is not able to predict price change when directly using raw price and volume history (A failed attempt is available at [this Google Colab notebook](https://drive.google.com/open?id=1PYPMb_-QsIgdmG4jEG-9ZuuxJMKuY4S2) and also included in this repo.
* The reason why LSTM is able to learn pattern from technical indicator features but not directly from raw price and volume history could be:
  * Preprocessing raw price and volume to technical indicators reduced the search space.
  * Some technical indicators such as SMA200 summarized a very long history. But when training on raw price and volume history, we did not use that long history due to concerns over the input dimension (because we have limited samples and there is this "curse of dimensionality" problem).
* The model does not generalize well to other assets. However, we may improve it by provide multiple training asset, each with a different characteristic.


