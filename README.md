[![Build Status](https://travis-ci.com/rubenbriones/Probabilistic-Sharpe-Ratio.svg?branch=master)](https://travis-ci.com/rubenbriones/Probabilistic-Sharpe-Ratio)
[![codecov](https://codecov.io/gh/rubenbriones/Probabilistic-Sharpe-Ratio/branch/master/graph/badge.svg)](https://codecov.io/gh/rubenbriones/Probabilistic-Sharpe-Ratio)

# Probabilistic-Sharpe-Ratio
Probabilistic Sharpe Ratio example in Python (by Marcos López de Prado)

### Link to the blog post with the complete explanation
https://quantdare.com/probabilistic-sharpe-ratio/

### Post summary
Imagine that we have one-year track-record of weekly returns from two different Hedge Funds in which we are interested to invest. So we have the last 52 weekly returns of both Hedge Funds. Let's look at their stats:

<p align="center">
  <img src="https://quantdare.com/wp-content/uploads/2020/05/Probabilistic-Sharpe-Ratio-2.png">
</p>

Mmm... it seems like the Hedge Fund 1 has a bigger Sharpe ratio, let's invest in it! **Wait a moment! What is about Probabilistic Sharpe Ratio, how confident can we be with our SR estimations?**

<p align="center">
  <img src="https://quantdare.com/wp-content/uploads/2020/05/Probabilistic-Sharpe-Ratio-3.png">
</p>

Ohh, now we can see that despite the bigger SR^ of the Hedge Fund 1 it seems more reasonable to invest our money in Hedge Fund 2! 

This is because we can not have the same confidence in our SR^ estimations. With HF1 we have "only" a certainty of 92.99% that in the future its true SR will be greater than 0 (SR*), but **with HF2 we have more statistically certainty that in the future will have positive returns: 95.19% of chances.**
