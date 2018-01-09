import csv
import numpy as np

def data_preprocessing():
    """
    """
    #loading data into stream
    stream = []
    with open('diamond-data.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, quotechar='|')
        next(reader)
        for row in reader:
            for i in range(len(row)):
                if (i == 0):
                    row[0] = (float(row[0])-0.2)/(5.01-0.2)
                if (i == 1):
                    row[1] = cut_conversion(row[1])
                if (i == 2):
                    row[2] = color_conversion(row[2])
                if (i == 3):
                    row[3] = clarity_conversion(row[3])
                if (i == 4):
                    row[4] = (float(row[4])-43)/(79-43)
                if (i == 5):
                    row[5] = (float(row[5])-43)/(95-43)
                if (i == 6):
                    row[6] = (float(row[6]))/(10.74)
                if (i == 7):
                    row[7] = (float(row[7]))/(58.9)
                if (i == 8):
                    row[8] = (float(row[8]))/(31.8)
                if (i == 9):
                    row[9] = (float(row[9])-326)/(18823-326)
            stream.append(row)
        outputFile = open("diamond-data-processed.csv", "w", newline='')
        with outputFile:
            writer = csv.writer(outputFile)
            writer.writerows(stream)
    
def cut_conversion(string):
    """
    Converts the text description of cut into numerical values.
    """
    if (string == 'Fair'):
        return 0.2
    if (string == 'Good'):
        return 0.4
    if (string == 'Very Good'):
        return 0.6
    if (string == 'Premium'):
        return 0.8
    if (string == 'Ideal'):
        return 1

def color_conversion(string):
    """
    Converts text description of color into numerical values.
    """
    if (string == 'J'):
        return 0.14
    if (string == 'I'):
        return 0.28
    if (string == 'H'):
        return 0.42
    if (string == 'G'):
        return 0.56
    if (string == 'F'):
        return 0.70
    if (string == 'E'):
        return 0.84
    if (string == 'D'):
        return 1
    
def clarity_conversion(string):
    """
    Converts text description of clarity into numerical values.
    """
    if (string == 'I1'):
        return 0.125
    if (string == 'SI2'):
        return 0.25
    if (string == 'SI1'):
        return 0.375
    if (string == 'VS2'):
        return 0.5
    if (string == 'VS1'):
        return 0.625
    if (string == 'VVS2'):
        return 0.75
    if (string == 'VVS1'):
        return 0.875
    if (string == 'IF'):
        return 1

def load_diamond_data():
    """
    """
    dataset = np.loadtxt('diamond-data-processed.csv', delimiter=",",skiprows=1)
    #there are 53940 data points
    train_data = dataset[0:32364]
    valid_data = dataset[32364:43152]
    test_data = dataset[43152:]
    
    X_train = train_data[:,0:8]
    Y_train = train_data[:,9]
    X_valid = train_data[:,0:8]
    Y_valid =train_data[:,9]
    X_test = test_data[:,0:8]
    Y_test = test_data[:,9]

    return (X_train,Y_train,X_valid,Y_valid,X_test,Y_test)