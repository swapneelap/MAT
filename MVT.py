import math
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

averageDays = 3600
percentage = 2.0/(1+averageDays)

################ FUNCTIONS ####################################

def Flip( frame ):
    "Flips the frame if not in past to present order."
    dateDelta = frame.at[0, 'Date'] - frame.at[1, 'Date']
    if dateDelta.days > 0:
        frame['Date'] = frame['Date'].values[::-1]
        frame['P/E'] = frame['P/E'].values[::-1]
        frame['P/B'] = frame['P/B'].values[::-1]

    return frame;

def DropNAN( frame ):
    "Removes the NAN in .csv"
    faults = np.array([])
    for index in range(0, frame.shape[0]):
        if math.isnan(frame.at[index, 'P/E']) or math.isnan(frame.at[index, 'P/B']):
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
                newRow = pd.DataFrame({'Date':(frame.at[index-1, 'Date'] + dt.timedelta(days=delta)), 'P/E':frame.at[index-1, 'P/E'], 'P/B':frame.at[index-1, 'P/B']}, index=[index+delta-1.5])
                frame = frame.append(newRow, ignore_index=False)
                frame = frame.sort_index().reset_index(drop=True)

    return frame;

def PE( frame ):
    zeros = [0.0] * frame.shape[0]
    frame['PEavg'] = zeros
    frame['PEpsdev'] = zeros
    frame['PEnsdev'] = zeros
    frame['PEpsdev2'] = zeros
    frame['PEnsdev2'] = zeros
    average = 0
    STD = 0

    for index in range(0, frame.shape[0]):
        group = np.array([])
        if index < averageDays-1:
            average = average + frame.at[index, 'P/E']
        elif index == averageDays-1:
            average = (average + frame.at[index, 'P/E'])/averageDays
            for subIndex in range(index-(averageDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/E']])
            STD = np.std(group)
            frame.at[index, 'PEavg'] = average
            frame.at[index, 'PEpsdev'] = average + STD
            frame.at[index, 'PEnsdev'] = average - STD
            frame.at[index, 'PEpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PEnsdev2'] = average - (2.0*STD)
        else:
            average = (frame.at[index, 'P/E'] * percentage) + (average * (1-percentage))
            for subIndex in range(index-(averageDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/E']])
            STD = np.std(group)
            frame.at[index, 'PEavg'] = average
            frame.at[index, 'PEpsdev'] = average + STD
            frame.at[index, 'PEnsdev'] = average - STD
            frame.at[index, 'PEpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PEnsdev2'] = average - (2.0*STD)
    return frame;

def PB( frame ):
    zeros = [0.0] * frame.shape[0]
    frame['PBavg'] = zeros
    frame['PBpsdev'] = zeros
    frame['PBnsdev'] = zeros
    frame['PBpsdev2'] = zeros
    frame['PBnsdev2'] = zeros
    average = 0
    STD = 0

    for index in range(0, frame.shape[0]):
        group = np.array([])
        if index < averageDays-1:
            average = average + frame.at[index, 'P/B']
        elif index == averageDays-1:
            average = (average + frame.at[index, 'P/B'])/averageDays
            for subIndex in range(index-(averageDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/B']])
            STD = np.std(group)
            frame.at[index, 'PBavg'] = average
            frame.at[index, 'PBpsdev'] = average + STD
            frame.at[index, 'PBnsdev'] = average - STD
            frame.at[index, 'PBpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PBnsdev2'] = average - (2.0*STD)
        else:
            average = (frame.at[index, 'P/B'] * percentage) + (average * (1-percentage))
            for subIndex in range(index-(averageDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/B']])
            STD = np.std(group)
            frame.at[index, 'PBavg'] = average
            frame.at[index, 'PBpsdev'] = average + STD
            frame.at[index, 'PBnsdev'] = average - STD
            frame.at[index, 'PBpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PBnsdev2'] = average - (2.0*STD)
    return frame;


###############################################################

fileName = input("Enter name of the csv file including .csv: ")

fileOpen = pd.read_csv(fileName)
dataFrame = pd.DataFrame(fileOpen, columns=["Date", "P/E", "P/B"])

dateFormat = input("Specify the date format(http://strftime.org/): ")

dataFrame.Date = pd.to_datetime(dataFrame.Date, format=dateFormat)

dataFrame = Flip(dataFrame)
dataFrame = DropNAN(dataFrame)
dataFrame = DataRefining(dataFrame)

dataFrame = PB(dataFrame)
print(dataFrame)

plt.plot(dataFrame['Date'], dataFrame['P/B'], 'k--', dataFrame['Date'], dataFrame['PBpsdev2'], dataFrame['Date'], dataFrame['PBpsdev'], dataFrame['Date'], dataFrame['PBavg'], dataFrame['Date'], dataFrame['PBnsdev'], dataFrame['Date'], dataFrame['PBnsdev2'])
plt.legend(['P/B', 'P/B 2STD', 'P/B STD', 'P/B avg', 'P/B -STD', 'P/B -2STD'])
plt.grid(True, linestyle='--')
plt.show()

'''
fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle(fileName, fontsize=12)
upperLimit = 80
lowerLimit = 40

axs[0].plot(dataFrame['Date'], dataFrame['RSI'], 'k', dataFrame['Date'], [upperLimit] * dataFrame.shape[0], 'r--', dataFrame['Date'], [lowerLimit] * dataFrame.shape[0], 'g--')
axs[0].legend(['RSI', 'Sell', 'Buy'], loc='upper left')
axs[0].grid(True, linestyle='--')

axs[1].plot(dataFrame['Date'], dataFrame['Close'], dataFrame['Date'], dataFrame['UpperBand'], 'r--', dataFrame['Date'], dataFrame['LowerBand'], 'g--')
axs[1].legend(['Close', 'Upper Bollinger band', 'Lower Bollinger band' ])
axs[1].grid(True, linestyle='--')

axs[2].plot(dataFrame['Date'], dataFrame['MACD'], 'b', dataFrame['Date'], dataFrame['MACDsignal'], 'r--', dataFrame['Date'], [0] * dataFrame.shape[0], 'k')
axs[2].legend(['MACD', 'Signal'])
axs[2].grid(True, linestyle='--')

plt.tight_layout()
plt.show()
'''
