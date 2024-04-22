import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def full_summary(df):
  #ignore image_floorplan
  try: 
    df_ = df.drop(columns=['image_floorplan'])
  except:
    pass

  descriptions = df_.describe()
  print('Data description:')
  display(descriptions)
  #Output as Latex 
  print(descriptions.to_latex())

  #Plot histograms of data 
  num_columns = df_.select_dtypes(include=['float64', 'int64']).columns
  for column in num_columns:
    plot_histogram(df_, column)
  
  #Heatmap 
  plot_heatmap(df_)

  #Do scatter plots over price and other columns
  for column in num_columns:
    plot_scatter(df_, column, 'price')


def plot_histogram(df, column):
  if column == 'price':
    #Set x-axis labels to be in half millions
    plt.hist(df[column], bins=20, range=(1000000, 9000000))


  else:
    plt.hist(df[column])
  plt.title(f'Histogram of {column}')
  plt.xlabel(column)
  plt.ylabel('Frequency')
  plt.show()

def plot_scatter(df, column1, column2):
  plt.scatter(df[column1], df[column2])
  plt.title(f'Scatter plot: {column1} vs {column2}')
  plt.xlabel(column1)
  plt.ylabel(column2)
  plt.show()

def plot_heatmap(df):
  plt.figure(figsize=(12, 10))
  #sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  #Show lower triangle
  corr = df.corr()
  matrix = np.triu(np.triu(np.ones_like(corr)))
  sns.heatmap(corr, annot=True, cmap='coolwarm', mask=matrix)
  plt.title('Correlation heatmap')
  plt.show()

