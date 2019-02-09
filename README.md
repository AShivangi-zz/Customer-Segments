# Unsupervised Learning
## Project: Creating Customer Segments

## Project Overview
In this project you will apply unsupervised learning techniques on product spending data 
collected for customers of a wholesale distributor in Lisbon, Portugal to identify customer 
segments hidden in the data. You will first explore the data by selecting a small subset to 
sample and determine if any product categories highly correlate with one another. Afterwards, 
you will preprocess the data by scaling each product category and then identifying (and removing) 
unwanted outliers. With the good, clean customer spending data, you will apply PCA 
transformations to the data and implement clustering algorithms to segment the transformed 
customer data. Finally, you will compare the segmentation found with an additional labeling 
and consider ways this information could assist the wholesale distributor with future service 
changes.

## Project Highlights

- Applied preprocessing techniques such as feature scaling and outlier detection.
- Interpreted data points that have been scaled, transformed, or reduced from PCA.
- Analyzed PCA dimensions and construct a new feature space.
- Otimally clusterd a set of data to find hidden patterns in a dataset.
- Assessed information given by cluster data and use it in a meaningful way.

## Software Requirements

This project uses the following software and Python libraries:

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

This project contains three files:

- `customer_segments.ipynb`: This is the main file.
- `customers.csv`: The project dataset.
- `visuals.py`: This Python script provides supplementary visualizations for the project.

## Running the Project
```
python customer_segments.py
```
