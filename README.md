# Solar Activity and Property Predictions using Deep Learning Methods

This repository is part of a research project for the course Advances in Deep Learning (2021-2022) at Leiden University. It contains deep learning methods for predicting the activity and magnetic properties on the surface of the Sun. Inspiration for this project was a [paper](https://arxiv.org/abs/1905.13575) and [GitHub repository](https://github.com/bionictoucan/Slic) by Armstrong et al., which presented a deep learning model for classifying images of Solar active regions. The project consists of three parts.

## Armstrong paper data analysis

The original direction of the project was to improve the classifier of Solar active region images. However, upon inspection of the data used by Armstrong et al., we found that the same active regions were included in both training and validation data. As the images from the same active region were generally similar, we verified that redistributing the training and validation data, such that active regions were exclusive to either set, did affect the performance significantly. We have included code for augmenting the original dataset, but ultimately, we shifted the focus of the project to different problems.

As volatile Solar activity can have hazardous consequences on Earth, it is useful to be able to predict such activity. This repository contains several models for predicting Solar activity.

## Active region property prediction

The 'AR_property_prediction' directory contains both an LSTM-based model and a Transformer-based model for solving the time-series-forecasting problem that is the prediction of the magnetic flux of an active region on the Sun. We predict the evolution of the magnetic flux 20 hours into the future, based on the evolution of several active region properties in the previous 40 hours.

## Flare prediction

In addition, the 'Flare_Prediction' directory also contains both and LSTM-based model and a Transformer-based model that predicts the occurrence of a Solar flare of a certain intensity within the next 24 hours. These models base their prediction on both active region properties and the Solar flare history in the last 10 hours.
