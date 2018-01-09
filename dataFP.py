import csv
import numpy as np


def load_wine_data():
    """
    Load the wine data from a csv file to data stream,
    then divide the data into training, validation, and
    test sets by 60%, 20%, and 20%, respectively.
    """
    data_stream = []
    with open('diamonds.csv', 'r', encoding='mbcs') as csvfile:
        reader = csv.reader(csvfile, quotechar='|')
        next(reader)
        for row in reader:
            data_stream.append(row)
           
    train_data = data_stream[0:43152] #60%
    valid_data = data_stream[43152:48546] #20%
    test_data = data_stream[48546:] #final 20%
    for i in range(len(train_data)):
        a = np.reshape(np.array(train_data[i][:6]), (6,1))
        if int(train_data[i][6]) > 12000:
            b = 0
        elif int(train_data[i][6]) > 6000:
            b = 1
        else:
            b = 2
        train_data[i] = (a, b)
    for j in range(len(valid_data)):
        x = np.array(valid_data[j][:6])
        y = np.reshape(x, (6,1))
        if int(valid_data[j][6]) > 12000:
            z = 0
        elif int(valid_data[j][6]) > 6000:
            z = 1
        else:
            z = 2
        valid_data[j] = (y, z)
    return (train_data, valid_data, test_data, data_stream)

"""
The rest of this file contains functions designed to
perform analytics on the data and guide development of
feature vector creation.
"""
def count_types(data_stream):
    """
    Counts the different number of types of countries,
    designations, provinces, region_1s, region_2s, varities,
    and wineries.
    """

    country_list = []
    designation_list = []
    province_list = []
    region_1_list = []
    region_2_list = []
    variety_list = []
    winery_list = []
    
    for entry in data_stream:
        
        if entry[1] not in country_list:
            country_list.append(entry[1])
        if entry[3] not in designation_list:
            designation_list.append(entry[3])
#        if entry[5] not in province_list:
#            province_list.append(entry[5])
#        if entry[6] not in region_1_list:
#            region_1_list.append(entry[6])
#        if entry[7] not in region_2_list:
#            region_2_list.append(entry[7])
#        if entry[8] not in variety_list:
#            variety_list.append(entry[8])
#        if entry[9] not in winery_list:
#            winery_list.append(entry[9])
    
    print('The # of countries is ' + str(len(country_list)))
    print('And they are: ' + str(country_list))
    print('The # of countries is ' + str(len(designation_list)))
    print('And they are: ' + str(designation_list))