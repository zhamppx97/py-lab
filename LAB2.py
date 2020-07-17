#IPython is what you are using now to run the notebook
import IPython
print("IPython version:      %6.6s (need at least 5.0.0)" % IPython.__version__)

# Numpy is a library for working with Arrays
import numpy as np
print("Numpy version:        %6.6s (need at least 1.12.0)" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print("SciPy version:        %6.6s (need at least 0.19.0)" % sp.__version__)

# Pandas makes working with data tables easier
import pandas as pd
print("Pandas version:       %6.6s (need at least 0.20.0)" % pd.__version__)

# Module for plotting
import matplotlib
print("Matplotlib version:    %6.6s (need at least 2.0.0)" % matplotlib.__version__)

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print("Scikit-Learn version: %6.6s (need at least 0.18.1)" % sklearn.__version__)

# Requests is a library for getting data from the Web
import requests
print("requests version:     %6.6s (need at least 2.9.0)" % requests.__version__)

#BeautifulSoup is a library to parse HTML and XML documents
import bs4
print("BeautifulSoup version:%6.6s (need at least 4.4)" % bs4.__version__)

import seaborn
print("Seaborn version:%6.6s (need at least 0.7)" % seaborn.__version__)
# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
# %matplotlib inline 
#this line above prepares the jupyter notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().
# notice we use short aliases here, and these are conventional in the python community

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm # allows us easy access to colormaps
import matplotlib.pyplot as plt # sets up plotting under plt
import pandas as pd #lets us handle data as dataframes

# sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options

# Hello matplotlib
x = np.linspace(0, 10, 30)  #array of 30 points from 0 to 10
y = np.sin(x)
z = y + np.random.normal(size=30) * .2
plt.plot(x, y, 'o-', label='A sine wave')
plt.plot(x, z, '-', label='Noisy sine')
plt.legend(loc = 'lower right')
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()

# Hello numpy
print("Make a 3 row x 4 column array of random numbers")
x = np.random.random((3, 4))
print(x,"\n")

print("Add 1 to every element")
x = x + 1
print(x,"\n")

print("Get the element at row 1, column 2")
print(x[1, 2])

# The colon syntax is called "slicing" the array. 
print("Get the first row")
print(x[0, :])

print("Last 2 items in the first row")
print(x[0, -2:])

print("Get every 2nd item in the first row")
print(x[0, ::2])

# Print the maximum, minimum, and mean of the array. 
# This does not require writing a loop. 
# In the code cell below, type x.m<TAB>, to find built-in operations for common array statistics like this
print("Max is  ", x.max())
print("Min is  ", x.min())
print("Mean is ", x.mean())

# Call the x.max function again, but use the axis keyword to print the maximum of each row in x.
print(x.max(axis=1))

# Here's a way to quickly simulate 500 coin "fair" coin tosses (where the probabily of getting Heads is 50%, or 0.5)
x = np.random.binomial(500, .5)
print("number of heads:", x)
# Repeat this simulation 500 times

# 3 ways to run the simulations

# loop
heads = []
for i in range(500):
    heads.append(np.random.binomial(500, .5))

# "list comprehension"
heads = [np.random.binomial(500, .5) for i in range(500)]

# pure numpy, preferred
heads = np.random.binomial(500, .5, size=500)
heads

# Use the plt.hist() function to plot a histogram of the number of Heads (1s) in each simulation
histogram = plt.hist(heads, bins=10)
plt.show()