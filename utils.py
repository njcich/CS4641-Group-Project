#### CS 4641 Group 2 Project helper functions
#### contributor: Zack Vogel (dvogel3

import pandas as pd



#########################################################
#helper functions for data cleaning, import, export, etc.
#########################################################

##load_csv function takes in a csv file path and loads it into pandas dataframe; fills in N/A values with 0
## Args: csv_file_path = file path as string to csv (i.e. "data.csv")
def load_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df.fillna(0)
    return df


### The export_to_csv function takes in a pandas dataframe and exports it to csv file in your local directory
## Args: df = pandas dataframe with data
##       file_name = the name of the csv file you wish to send the dataframe to (i.e. "data.csv");
##                   sends to local directory
##       index     = row names as a list of strings; default is None
##       header    = column names; default is True that column names exist
def export_to_csv(df, file_name, index=None, header = True):
    export = df.to_csv(file_name, index, header)
    print("Export to " + file_name + " Complete")


## Example visualization of pandas dataframe with histogram; histogram based on columns
## Args: df = pandas dataframe
##       bins = number of bins for histogram
##       cols = list of column labels you want to visualize
##       Will plot subplots of different histograms per column label
def viz_histogram(df, bins, cols):

    df.columns[cols].diff.hist(bins)

    print("Visualization Complete")