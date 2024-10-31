# spaceflights

## Overview

The spaceflights project is a ML/DL project provided by Kedro to work with their [great tutorial] (https://www.youtube.com/playlist?list=PL-JJgymPjK5LddZXbIzp9LWurkLGgB-nY).
As you can read in the [Kedro documentation] (https://docs.kedro.org/en/stable/tutorial/spaceflights_tutorial.html), in this project "You want to construct a model that predicts the price for each trip to the Moon and the corresponding return flight."

## Action Plan

In the tutorial, they constructed a linear regression model using scikit-learn. They experimented with 2 different sets of model's features using pipeline namespaces. Overall, the model performance isn't great, it achieves r2 score in the range of (0.2 - 0.4).

The next steps I will take will be:

* Make use of the features they didn't use (some features have to be hot-encoded first)
* Try multi-layer dense model
* Try Ridge and Lasso Regression
* Try decision tree
* Try random forest