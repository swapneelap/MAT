from pandas_datareader import data as pdr
import yfinance as yf
import math
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

daysFullAvg = 200
daysHalfAvg = 50
signalDays = 30
strengthDays = 14
RSIavgDays = 9
RSImatureDays = 260

################ CLASS ########################################

class SnaptoCursor(object):
    def __init__(self, ax, frame):
        self.ax = ax
        self.ly = ax.axvline(color='k')  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", linewidth=5, color="orange", zorder=5) 
        self.frame = frame
        self.txt = ax.text(0.07, 0.09, '', bbox={'facecolor':'white', 'alpha':0.7, 'pad':2}, fontsize=11)

    def mouse_move(self, event):
        if not event.inaxes: return
        x = matplotlib.dates.num2date(event.xdata)
        x = x.strftime('%Y-%m-%d')
        indx = dataFrame.index[dataFrame.Date == x]
        x = self.frame.at[indx[0], 'Date']
        y = self.frame.at[indx[0], 'Close']
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text(math.trunc(y))
        self.txt.set_position((x+dt.timedelta(days=5), y+10))
        self.ax.figure.canvas.draw_idle()




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
    fullAvg = 0
    percentage = 2.0/(daysFullAvg+1)
    for index in range(0, frame.shape[0]):
        if index < daysFullAvg-1:
            fullAvg = fullAvg + frame.at[index, 'Close']
        elif index == daysFullAvg-1:
            fullAvg = (fullAvg + frame.at[index, 'Close'])/daysFullAvg
            frame.at[index, 'FullAvg'] = fullAvg
        else:
            fullAvg = (frame.at[index, 'Close'] * percentage) + (fullAvg * (1-percentage))
            frame.at[index, 'FullAvg'] = fullAvg

    return frame;

def HalfAverage( frame ):
    "Half exponential moving average"
    zeros = [0.0] * frame.shape[0]
    frame['HalfAvg'] = zeros
    halfAvg = 0
    percentage = 2.0/(daysHalfAvg+1)
    for index in range(0, frame.shape[0]):
        if index < daysHalfAvg-1:
            halfAvg = halfAvg + frame.at[index, 'Close']
        elif index == daysHalfAvg-1:
            halfAvg = (halfAvg + frame.at[index, 'Close'])/daysHalfAvg
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
    daysMACD = daysFullAvg
    daysMACDsig = daysMACD + signalDays
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
    days = daysHalfAvg

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
    days = RSIavgDays
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

compnayName = input("Enter yfinance Symbol for the compnay: ")
startDate = input("Enter the start date as yyyy-mm-dd: ")
endDate = input("Enter the end date as yyyy-mm-dd: ")

rawData = pdr.get_data_yahoo(compnayName, start=startDate, end=endDate)
dataFrame = pd.DataFrame(rawData, columns=["Close"])
dataFrame.reset_index(level=['Date'], inplace=True)
dataFrame.Date = pd.to_datetime(dataFrame.Date, format='%Y-%m-%d')

dataFrame = Flip(dataFrame)
dataFrame = DropNAN(dataFrame)
dataFrame = DataRefining(dataFrame)

dataFrame = FullAverage(dataFrame)
dataFrame = HalfAverage(dataFrame)
dataFrame = MACD(dataFrame)
dataFrame = BollingerBands(dataFrame)
dataFrame = RSI(dataFrame)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle(compnayName, fontsize=12)
upperLimit = 80
lowerLimit = 40

axs[0].plot(dataFrame['Date'], [upperLimit] * dataFrame.shape[0], 'r--', dataFrame['Date'], [lowerLimit] * dataFrame.shape[0], 'g--', dataFrame['Date'], dataFrame['RSI'], dataFrame['Date'], dataFrame['RSIavg'], 'k--')
axs[0].legend(['Sell', 'Buy', 'RSI', 'RSI average'], loc='upper left')
axs[0].grid(True, linestyle='--')

axs[1].plot(dataFrame['Date'], dataFrame['Close'], dataFrame['Date'], dataFrame['UpperBand'], 'r--', dataFrame['Date'], dataFrame['LowerBand'], 'g--')
axs[1].legend(['Close', 'Upper Bollinger band', 'Lower Bollinger band' ])
axs[1].set_ylim([dataFrame['Close'].min(), dataFrame['UpperBand'].max()])
axs[1].grid(True, linestyle='--')

cursor = SnaptoCursor(axs[1], dataFrame)
cid =  plt.connect('motion_notify_event', cursor.mouse_move)

axs[2].plot(dataFrame['Date'], dataFrame['MACD'], 'b', dataFrame['Date'], dataFrame['MACDsignal'], 'r--', dataFrame['Date'], [0] * dataFrame.shape[0], 'k')
axs[2].legend(['MACD', 'Signal'])
axs[2].grid(True, linestyle='--')

plt.xlim(dataFrame.at[0, 'Date'], dataFrame.at[dataFrame.shape[0]-1, 'Date'])
plt.tight_layout()
plt.show()
