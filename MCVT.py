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
valuationDays = 3600
chartingSwitch = False
valuationSwitch = False
################Charting FUNCTIONS ####################################

def Flip_CH( frame ):
    "Flips the frame if not in past to present order."
    dateDelta = frame.at[0, 'Date'] - frame.at[1, 'Date']
    if dateDelta.days > 0:
        frame['Date'] = frame['Date'].values[::-1]
        frame['Close'] = frame['Close'].values[::-1]
        print('Flipping the dataFrame')

    return frame;

def DropNAN_CH( frame ):
    "Removes the NAN in .csv"
    faults = np.array([])
    for index in range(0, frame.shape[0]):
        if math.isnan(frame.at[index, 'Close']):
            faults = np.append(faults, [index])

    print ("Removing NANs at ", faults)
    frame = frame.drop(faults)
    frame = frame.sort_index().reset_index(drop=True)
    return frame;

def DataRefining_CH( frame ):
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
    days = 9
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
################ Valuation Functions ##########################
def Flip_VA( frame ):
    "Flips the frame if not in past to present order."
    dateDelta = frame.at[0, 'Date'] - frame.at[1, 'Date']
    if dateDelta.days > 0:
        frame['Date'] = frame['Date'].values[::-1]
        frame['P/E'] = frame['P/E'].values[::-1]
        frame['P/B'] = frame['P/B'].values[::-1]

        print("Flipping the dataFrame")

    return frame;

def DropNAN_VA( frame ):
    "Removes the NAN in .csv"
    faults = np.array([])
    for index in range(0, frame.shape[0]):
        if math.isnan(frame.at[index, 'P/E']) or math.isnan(frame.at[index, 'P/B']):
            faults = np.append(faults, [index])

    print ("Removing NANs at ", faults)
    frame = frame.drop(faults)
    frame = frame.sort_index().reset_index(drop=True)
    return frame;

def DataRefining_VA( frame ):
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
    percentage = 2.0/(1+valuationDays)

    for index in range(0, frame.shape[0]):
        group = np.array([])
        if index < valuationDays-1:
            average = average + frame.at[index, 'P/E']
        elif index == valuationDays-1:
            average = (average + frame.at[index, 'P/E'])/valuationDays
            for subIndex in range(index-(valuationDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/E']])
            STD = np.std(group)
            frame.at[index, 'PEavg'] = average
            frame.at[index, 'PEpsdev'] = average + STD
            frame.at[index, 'PEnsdev'] = average - STD
            frame.at[index, 'PEpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PEnsdev2'] = average - (2.0*STD)
        else:
            average = (frame.at[index, 'P/E'] * percentage) + (average * (1-percentage))
            for subIndex in range(index-(valuationDays-1), index+1):
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
    percentage = 2.0/(valuationDays)

    for index in range(0, frame.shape[0]):
        group = np.array([])
        if index < valuationDays-1:
            average = average + frame.at[index, 'P/B']
        elif index == valuationDays-1:
            average = (average + frame.at[index, 'P/B'])/valuationDays
            for subIndex in range(index-(valuationDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/B']])
            STD = np.std(group)
            frame.at[index, 'PBavg'] = average
            frame.at[index, 'PBpsdev'] = average + STD
            frame.at[index, 'PBnsdev'] = average - STD
            frame.at[index, 'PBpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PBnsdev2'] = average - (2.0*STD)
        else:
            average = (frame.at[index, 'P/B'] * percentage) + (average * (1-percentage))
            for subIndex in range(index-(valuationDays-1), index+1):
                group = np.append(group, [frame.at[subIndex, 'P/B']])
            STD = np.std(group)
            frame.at[index, 'PBavg'] = average
            frame.at[index, 'PBpsdev'] = average + STD
            frame.at[index, 'PBnsdev'] = average - STD
            frame.at[index, 'PBpsdev2'] = average + (2.0*STD)
            frame.at[index, 'PBnsdev2'] = average - (2.0*STD)
    return frame;

def ROE( frame ):
    zeros = [0.0] * frame.shape[0]
    frame['ROE'] = zeros
    for index in range(0, frame.shape[0]):
        frame.at[index, 'ROE'] = frame.at[index, 'P/B']/frame.at[index, 'P/E']
    return frame;
###############################################################

while (True):
    answer = input("Display charts? [y/n] ")
    if answer == "y":
        chartingSwitch = True
        break
    elif answer == "n":
        break
    else:
        print("Please answer y or n")

while (True):
    answer = input("Display valuation? [y/n] ")
    if answer == "y":
        valuationSwitch = True
        break
    elif answer == 'n':
        break
    else:
        print("Please answer y or n")

if chartingSwitch:
    fileName_CH = input("Enter name of the csv file with 'Close' price including .csv: ")

    fileOpen_CH = pd.read_csv(fileName_CH)
    dataFrame_CH = pd.DataFrame(fileOpen_CH, columns=["Date", "Close"])

    dateFormat_CH = input("Specify the date format(http://strftime.org/): ")

    dataFrame_CH.Date = pd.to_datetime(dataFrame_CH.Date, format=dateFormat_CH)

    dataFrame_CH = Flip_CH(dataFrame_CH)
    dataFrame_CH = DropNAN_CH(dataFrame_CH)
    dataFrame_CH = DataRefining_CH(dataFrame_CH)

    dataFrame_CH = FullAverage(dataFrame_CH)
    dataFrame_CH = HalfAverage(dataFrame_CH)
    dataFrame_CH = MACD(dataFrame_CH)
    dataFrame_CH = BollingerBands(dataFrame_CH)
    dataFrame_CH = RSI(dataFrame_CH)

    fig1, axs1 = plt.subplots(3, 1, sharex=True)
    fig1.suptitle(fileName_CH, fontsize=12)
    upperLimit = 80
    lowerLimit = 40

    axs1[0].plot(dataFrame_CH['Date'], [upperLimit] * dataFrame_CH.shape[0], 'r--', dataFrame_CH['Date'], [lowerLimit] * dataFrame_CH.shape[0], 'g--', dataFrame_CH['Date'], dataFrame_CH['RSI'], dataFrame_CH['Date'], dataFrame_CH['RSIavg'], 'k--')
    axs1[0].legend(['Sell', 'Buy', 'RSI', 'RSI average(9)'], loc='upper left')
    axs1[0].grid(True, linestyle='--')

    axs1[1].plot(dataFrame_CH['Date'], dataFrame_CH['Close'], dataFrame_CH['Date'], dataFrame_CH['UpperBand'], 'r--', dataFrame_CH['Date'], dataFrame_CH['LowerBand'], 'g--')
    axs1[1].legend(['Close', 'Upper Bollinger band', 'Lower Bollinger band' ])
    axs1[1].grid(True, linestyle='--')

    axs1[2].plot(dataFrame_CH['Date'], dataFrame_CH['MACD'], 'b', dataFrame_CH['Date'], dataFrame_CH['MACDsignal'], 'r--', dataFrame_CH['Date'], [0] * dataFrame_CH.shape[0], 'k')
    axs1[2].legend(['MACD', 'Signal'])
    axs1[2].grid(True, linestyle='--')

    fig1.tight_layout()

if valuationSwitch:
    fileName_VA = input("Enter name of the csv file with 'PE PB'ratio including .csv: ")

    fileOpen_VA = pd.read_csv(fileName_VA)
    dataFrame_VA = pd.DataFrame(fileOpen_VA, columns=["Date", "P/E", "P/B"])

    dateFormat_VA = input("Specify the date format(http://strftime.org/): ")

    dataFrame_VA.Date = pd.to_datetime(dataFrame_VA.Date, format=dateFormat_VA)

    dataFrame_VA = Flip_VA(dataFrame_VA)
    dataFrame_VA = DropNAN_VA(dataFrame_VA)
    dataFrame_VA = DataRefining_VA(dataFrame_VA)

    dataFrame_VA = PE(dataFrame_VA)
    dataFrame_VA = PB(dataFrame_VA)
    dataFrame_VA = ROE(dataFrame_VA)

    fig2, axs2 = plt.subplots(3, 1, sharex=True)
    fig2.suptitle(fileName_VA, fontsize=12)

    axs2[0].plot(dataFrame_VA['Date'], dataFrame_VA['P/E'], '--', dataFrame_VA['Date'], dataFrame_VA['PEavg'], dataFrame_VA['Date'], dataFrame_VA['PEpsdev2'], dataFrame_VA['Date'], dataFrame_VA['PEpsdev'], dataFrame_VA['Date'], dataFrame_VA['PEnsdev'], dataFrame_VA['Date'], dataFrame_VA['PEnsdev2'])
    axs2[0].legend(['P/E', 'P/E avg', '+2STD', '+STD', '-STD', '-2STD'])
    axs2[0].grid(True, linestyle='--')

    axs2[1].plot(dataFrame_VA['Date'], dataFrame_VA['P/B'], '--', dataFrame_VA['Date'], dataFrame_VA['PBavg'], dataFrame_VA['Date'], dataFrame_VA['PBpsdev2'], dataFrame_VA['Date'], dataFrame_VA['PBpsdev'], dataFrame_VA['Date'], dataFrame_VA['PBnsdev'], dataFrame_VA['Date'], dataFrame_VA['PBnsdev2'])
    axs2[1].legend(['P/B', 'P/B avg', '+2STD', '+STD', '-STD', '-2STD'])
    axs2[1].grid(True, linestyle='--')

    axs2[2].plot(dataFrame_VA['Date'], dataFrame_VA['ROE'], 'b')
    axs2[2].legend(['ROE'])
    axs2[2].grid(True, linestyle='--')

    fig2.tight_layout()

if chartingSwitch or valuationSwitch:
    plt.show()
