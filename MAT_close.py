import math
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

averageDays = 360
strengthDays = 14

################ FUNCTIONS ####################################

def Flip( frame ):
    "Flips the frame if not in past to present order."
    dateDelta = frame.at[0, 'Date'] - frame.at[1, 'Date']
    if dateDelta.days > 0:
        frame['Date'] = frame['Date'].values[::-1]
        frame['Close'] = frame['Close'].values[::-1]

    return frame;

def DropNAN( frame ):
    "Removes the NAN in .csv"
    faults = np.array([])
    for index in range(0, frame.shape[0]):
        if math.isnan(frame.at[index, 'Close']):
            faults = np.append(faults, [index])

    print ("Removing NANs at ", faults)
    frame = frame.drop(faults)
    frame = frame.sort_index().reset_index(drop=True)
    return frame;

def DataRefining( frame ):
    "Adds the missing days' Close valus to make the data smooth"
    totalDays = frame.at[frame.shape[0]-1, 'Date'] - frame.at[0, 'Date']
    for index in range(1, totalDays.days+1):

        dateDelta = frame.at[index, 'Date'] - frame.at[index-1, 'Date']

        if dateDelta.days > 1:
            for delta in range(1, dateDelta.days):
                newRow = pd.DataFrame({'Date':(frame.at[index-1, 'Date'] + dt.timedelta(days=delta)), 'Close':frame.at[index-1, 'Close']}, index=[index+delta-1.5])
                frame = frame.append(newRow, ignore_index=False)
                frame = frame.sort_index().reset_index(drop=True)

    return frame;

def FullAverage( frame ):
    "Full exponential moving average"
    zeros = [0.0] * frame.shape[0]
    frame['FullAvg'] = zeros
    days = averageDays
    fullAvg = 0
    percentage = 2.0/(days+1)
    for index in range(0, frame.shape[0]):
        if index < days-1:
            fullAvg = fullAvg + frame.at[index, 'Close']
        elif index == days-1:
            fullAvg = (fullAvg + frame.at[index, 'Close'])/days
            frame.at[index, 'FullAvg'] = fullAvg
        else:
            fullAvg = (frame.at[index, 'Close'] * percentage) + (fullAvg * (1-percentage))
            frame.at[index, 'FullAvg'] = fullAvg

    return frame;

def HalfAverage( frame ):
    "Half exponential moving average"
    zeros = [0.0] * frame.shape[0]
    frame['HalfAvg'] = zeros
    days = math.floor(averageDays/2.0)
    halfAvg = 0
    percentage = 2.0/(days+1)
    for index in range(0, frame.shape[0]):
        if index < days-1:
            halfAvg = halfAvg + frame.at[index, 'Close']
        elif index == days-1:
            halfAvg = (halfAvg + frame.at[index, 'Close'])/days
            frame.at[index, 'HalfAvg'] = halfAvg
        else:
            halfAvg = (frame.at[index, 'Close'] * percentage) + (halfAvg * (1-percentage))
            frame.at[index, 'HalfAvg'] = halfAvg

    return frame;

def MACD( frame ):
    "Moving Average Convergence Divergence"
    zeros = [0.0] * frame.shape[0]
    frame['MACD'] = zeros
    frame['MACDsignal'] = zeros
    daysMACD = averageDays
    daysMACDsig = averageDays+130
    macdSignal = 0
    percentage = 2.0/(130+1)

    for index in range(daysMACD-1, frame.shape[0]):
        frame.at[index, 'MACD'] = frame.at[index, 'HalfAvg'] - frame.at[index, 'FullAvg']
        if index < daysMACDsig-1:
            macdSignal = macdSignal + frame.at[index, 'MACD']
        elif index == daysMACDsig-1:
            macdSignal = (macdSignal + frame.at[index, 'MACD'])/130
            frame.at[index, 'MACDsignal'] = macdSignal
        else:
            macdSignal = (frame.at[index, 'MACD'] * percentage) + (macdSignal * (1-percentage))
            frame.at[index, 'MACDsignal'] = macdSignal

    return frame;

def BollingerBands( frame ):
    zeros = [0.0] * frame.shape[0]
    frame['UpperBand'] = zeros
    frame['LowerBand'] = zeros
    STD = 0
    days = math.floor(averageDays/2.0)

    for index in range(days-1, frame.shape[0]):
        group = np.array([])
        for subIndex in range(index-(days-1), index+1):
            group = np.append(group, [frame.at[subIndex, 'Close']])
        STD = np.std(group)
        frame.at[index, 'UpperBand'] = frame.at[index, 'HalfAvg'] + (2.0*STD)
        frame.at[index, 'LowerBand'] = frame.at[index, 'HalfAvg'] - (2.0*STD)

    return frame;

###############################################################

fileName = input("Enter name of the csv file including .csv: ")

fileOpen = pd.read_csv(fileName)
dataFrame = pd.DataFrame(fileOpen, columns=["Date", "Close"])

dateFormat = input("Specify the date format(http://strftime.org/): ")

dataFrame.Date = pd.to_datetime(dataFrame.Date, format=dateFormat)

dataFrame = Flip(dataFrame)
dataFrame = DropNAN(dataFrame)
dataFrame = DataRefining(dataFrame)

dataFrame = FullAverage(dataFrame)
dataFrame = HalfAverage(dataFrame)
dataFrame = MACD(dataFrame)
dataFrame = BollingerBands(dataFrame)

plt.plot(dataFrame['Date'], dataFrame['Close'], dataFrame['Date'], dataFrame['UpperBand'], dataFrame['Date'], dataFrame['LowerBand'])
plt.legend(['Close', 'Upper', 'Lower'])
plt.show()
