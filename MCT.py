import math
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

averageDays = 360
signalDays = 130
strengthDays = 14
RSImatureDays = 260

################ FUNCTIONS ####################################

def Flip( frame ):
    "Flips the frame if not in past to present order."
    dateDelta = frame.at[0, 'Date'] - frame.at[1, 'Date']
    if dateDelta.days > 0:
        frame['Date'] = frame['Date'].values[::-1]
        frame['Close'] = frame['Close'].values[::-1]
        print('Flipping the dataFrame')

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
    daysMACDsig = averageDays+signalDays
    macdSignal = 0
    percentage = 2.0/(signalDays+1)

    for index in range(daysMACD-1, frame.shape[0]):
        frame.at[index, 'MACD'] = frame.at[index, 'HalfAvg'] - frame.at[index, 'FullAvg']
        if index < daysMACDsig-1:
            macdSignal = macdSignal + frame.at[index, 'MACD']
        elif index == daysMACDsig-1:
            macdSignal = (macdSignal + frame.at[index, 'MACD'])/signalDays
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

def RSI( frame ):
    "Relative Strength Indicator"
    zeros = [0.0] * frame.shape[0]
    frame['RSI'] = zeros

    averageGain = 0
    averageLoss = 0

    for index in range(1, frame.shape[0]):
        signedDiff = frame.at[index, 'Close'] - frame.at[index-1, 'Close']
        priceDiff = abs(signedDiff)
        if index < strengthDays-1:
            if signedDiff >= 0:
                averageGain = averageGain + priceDiff
            else:
                averageLoss = averageLoss + priceDiff
        elif index == strengthDays-1:
            if signedDiff >=0:
                averageGain = (averageGain + priceDiff)/strengthDays
                averageLoss = averageLoss/strengthDays
                frame.at[index, 'RSI'] = 100.0 - (100/(1+(averageGain/averageLoss)))
            else:
                averageGain = averageGain/strengthDays
                averageLoss = (averageLoss + priceDiff)/strengthDays
                frame.at[index, 'RSI'] = 100.0 - (100/(1+(averageGain/averageLoss)))
        else:
            if signedDiff >=0:
                averageGain = ((averageGain * (strengthDays-1))+priceDiff)/strengthDays
                averageLoss = averageLoss * (strengthDays-1)/strengthDays
                frame.at[index, 'RSI'] = 100.0 - (100/(1+(averageGain/averageLoss)))
            else:
                averageGain = averageGain * (strengthDays-1)/strengthDays
                averageLoss = ((averageLoss * (strengthDays-1))+priceDiff)/strengthDays
                frame.at[index, 'RSI'] = 100.0 - (100/(1+(averageGain/averageLoss)))

    frame['RSIavg'] = zeros
    days = 7
    rsiAverage = 0
    percentage = 2.0/(1+days)
    for index in range((strengthDays-1)+RSImatureDays, frame.shape[0]):
        if index < (strengthDays-1)+RSImatureDays+days-1:
            rsiAverage = rsiAverage + frame.at[index, 'RSI']
        elif index == (strengthDays-1)+RSImatureDays+days-1:
            rsiAverage = (rsiAverage+frame.at[index, 'RSI'])/days
            frame.at[index, 'RSIavg'] = rsiAverage
        else:
            rsiAverage = (frame.at[index, 'RSI'] * percentage) + (rsiAverage * (1-percentage))
            frame.at[index, 'RSIavg'] = rsiAverage

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
dataFrame = RSI(dataFrame)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle(fileName, fontsize=12)
upperLimit = 80
lowerLimit = 40

axs[0].plot(dataFrame['Date'], [upperLimit] * dataFrame.shape[0], 'r--', dataFrame['Date'], [lowerLimit] * dataFrame.shape[0], 'g--', dataFrame['Date'], dataFrame['RSIavg'])
axs[0].legend(['Sell', 'Buy', 'RSI week average'], loc='upper left')
axs[0].grid(True, linestyle='--')

axs[1].plot(dataFrame['Date'], dataFrame['Close'], dataFrame['Date'], dataFrame['UpperBand'], 'r--', dataFrame['Date'], dataFrame['LowerBand'], 'g--')
axs[1].legend(['Close', 'Upper Bollinger band', 'Lower Bollinger band' ])
axs[1].grid(True, linestyle='--')

axs[2].plot(dataFrame['Date'], dataFrame['MACD'], 'b', dataFrame['Date'], dataFrame['MACDsignal'], 'r--', dataFrame['Date'], [0] * dataFrame.shape[0], 'k')
axs[2].legend(['MACD', 'Signal'])
axs[2].grid(True, linestyle='--')

plt.tight_layout()
plt.show()
