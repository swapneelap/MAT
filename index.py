#!/usr/bin/python2
import numpy as np
import datetime as dt
import pandas as pd
import math
import matplotlib.pyplot as plt

test = pd.read_csv('britannia.csv')
final = pd.DataFrame(test, columns=["Date", "Close"])

#final["Date"] = final["Date"].values[::-1]
#final["Close"] = final["Close"].values[::-1]

#final.Date = pd.to_datetime(final.Date, format='%d %b %Y')
final.Date = pd.to_datetime(final.Date, format='%Y-%m-%d')
###### dropping NAN values ###################################
fault = np.array([])
for indx in range(0, final.shape[0]):
    if math.isnan(final.at[indx, 'Close']):
        fault = np.append(fault, [indx])

print(fault)
final = final.drop(fault)

##### Making the data uniform ################################

length = final.shape[0]
numDays = 0
totalDays = final.at[final.shape[0]-1, 'Date'] - final.at[0, 'Date']

for indx in range(1, totalDays.days):
    dateDelta = final.at[indx, 'Date'] - final.at[indx-1, 'Date']
    if dateDelta.days > 1:
        for delta in range(1,dateDelta.days):
            newRow = pd.DataFrame({'Date': final.at[indx-1, 'Date'] + dt.timedelta(days=delta), 'Close': final.at[indx-1, 'Close']}, index=[indx+delta-1.5])
            final = final.append(newRow, ignore_index=False)
            final = final.sort_index().reset_index(drop=True)

'''
#############################################################
############### averaging 180 days #########################

zeros = [0.0] * final.shape[0]
final['Avg180'] = zeros
avg180 = 0
for indx in range(0,final.shape[0]):
    if indx < 179:
        avg180 = avg180 + final.at[indx, 'Close']
    elif indx == 179:
        avg180 = avg180/180
        final.at[indx, 'Avg180'] = avg180
    else:
        avg180 = avg180 + (final.at[indx, 'Close'] - final.at[(indx-180), 'Close'])/180
        final.at[indx, 'Avg180'] = avg180


#####################################################################
################## averaging 360 days ###############################

zeros = [0.0] * final.shape[0]
final['Avg360'] = zeros
avg360 = 0
for indx in range(0,final.shape[0]):
    if indx < 359:
        avg360 = avg360 + final.at[indx, 'Close']
    elif indx == 359:
        avg360 = avg360/360
        final.at[indx, 'Avg360'] = avg360
    else:
        avg360 = avg360 + (final.at[indx, 'Close'] - final.at[(indx-360), 'Close'])/360
        final.at[indx, 'Avg360'] = avg360




ax = plt.gca()
final.plot(kind='line', x='Date', y='Close', ax=ax)
final.plot(kind='line', x='Date', y='Avg180', ax=ax)
final.plot(kind='line', x='Date', y='Avg360', ax=ax)
plt.xticks(rotation=45)
plt.title("NIFTY smallcap 250")
plt.show()

#########################################################################
################## expo avg 180 days ####################################

zeros = [0.0] * final.shape[0]
final['ExAvg180'] = zeros
exavg180 = 0
for indx in range(0,final.shape[0]):
    if indx < 179:
        exavg180 = exavg180 + final.at[indx, 'Close']
    elif indx == 179:
        exavg180 = (exavg180+final.at[indx, 'Close'])/180
        final.at[indx, 'ExAvg180'] = exavg180
    else:
        perc = 2.0/(180+1)
        exavg180 = (final.at[indx, 'Close'] * perc) + (exavg180 * (1-perc))
        final.at[indx, 'ExAvg180'] = exavg180

##############################################################################
######################## Ex avg 360 days #####################################

zeros = [0.0] * final.shape[0]
final['ExAvg360'] = zeros
exavg360 = 0
for indx in range(0,final.shape[0]):
    if indx < 359:
        exavg360 = exavg360 + final.at[indx, 'Close']
    elif indx == 359:
        exavg360 = (exavg360+final.at[indx, 'Close'])/360
        final.at[indx, 'ExAvg360'] = exavg360
    else:
        perc = 2.0/(360+1)
        exavg360 = (final.at[indx, 'Close'] * perc) + (exavg360 * (1-perc))
        final.at[indx, 'ExAvg360'] = exavg360


###############################################################################
###################### Moving Average Convergence Divergence ##################

zeros = [0.0] * final.shape[0]
final['MACD'] = zeros
final['MACD_signal'] = zeros

for indx in range(0,final.shape[0]):
    if indx < 359:
        final.at[indx, 'MACD'] = 0
    else:
        final.at[indx, 'MACD'] = final.at[indx, 'ExAvg180'] - final.at[indx, 'ExAvg360']

macdSig = 0
for indx in range(0,final.shape[0]):
    if indx < 489:
        macdSig = macdSig + final.at[indx, 'MACD']
    elif indx == 489:
        macdSig = (macdSig+final.at[indx, 'MACD'])/130
        final.at[indx, 'MACD_signal'] = macdSig
    else:
        perc = 2.0/(130+1)
        macdSig = (final.at[indx, 'MACD'] * perc) + (macdSig * (1-perc))
        final.at[indx, 'MACD_signal'] = macdSig

zeros = [0.0] * final.shape[0]
final['MACD_diff'] = zeros

for indx in range(489, final.shape[0]):
    final.at[indx, 'MACD_diff'] = (final.at[indx, 'MACD'] - final.at[indx, 'MACD_signal'])


plt.subplot(212)
plt.plot(final['Date'], final['MACD'], 'b-', final['Date'], final['MACD_signal'], 'r--')
plt.legend(['MACD', 'Signal'])
plt.axhline(color='k')
'''
##############################################################################
################### Bollinger Bands ##########################################

zeros = [0.0] * final.shape[0]
final['Avg200'] = zeros
final['Upper_band'] = zeros
final['Lower_band'] = zeros
avg200 = 0

for indx in range(0, final.shape[0]):
    if indx < 199:
        avg200 = avg200 + final.at[indx, 'Close']
    elif indx == 199:
        avg200 = (avg200+final.at[indx, 'Close'])/200
        std = 0
        group = np.array([])
        for subindx in range(0,indx+1):
            group = np.append(group, [final.at[subindx, 'Close']])
        std = np.std(group)
        final.at[indx, 'Avg200'] = avg200
        final.at[indx, 'Upper_band'] = avg200 + (2.0*std)
        final.at[indx, 'Lower_band'] = avg200 - (2.0*std)
    else:
        avg200 = avg200 + (final.at[indx, 'Close'] - final.at[(indx-200), 'Close'])/200
        std = 0
        group = np.array([])
        for subindx in range(indx-199,indx+1):
            group = np.append(group, [final.at[subindx, 'Close']])
        std = np.std(group)
        final.at[indx, 'Avg200'] = avg200
        final.at[indx, 'Upper_band'] = avg200 + (2.0*std)
        final.at[indx, 'Lower_band'] = avg200 - (2.0*std)
plt.subplot(211)
plt.plot(final['Date'], final['Upper_band'], 'r--', final['Date'], final['Close'], '-', final['Date'], final['Lower_band'], 'g--')
plt.grid(True, linestyle='--')
plt.legend(['Up', 'Close', 'Low'])
#plt.show()

###############################################################################
####################### Relative Strength indicator ###########################
zeros = [0.0] * final.shape[0]
final['RSI'] = zeros
avgGain = 0
avgLoss = 0
rsiDays = 14

for indx in range(1, final.shape[0]):
    diff = final.at[indx, 'Close'] - final.at[indx-1, 'Close']
    if indx < (rsiDays-1):
        if diff >= 0:
            avgGain = avgGain + diff
        else:
            avgLoss = avgLoss + abs(diff)
    elif indx == (rsiDays-1):
        if diff >= 0:
            avgGain = (avgGain + diff)/rsiDays
            avgLoss = avgLoss/rsiDays
            final.at[indx, 'RSI'] = 100.0 - (100/(1+(avgGain/avgLoss)))
        else:
            avgGain = avgGain/rsiDays
            avgLoss = (avgLoss+abs(diff))/rsiDays
            final.at[indx, 'RSI'] = 100.0 - (100/(1+(avgGain/avgLoss)))
    else:
        if diff >= 0:
            avgGain = ((avgGain * (rsiDays-1)) + diff)/rsiDays
            avgLoss = avgLoss * (rsiDays-1)/rsiDays
            final.at[indx, 'RSI'] = 100.0 - (100/(1+(avgGain/avgLoss)))
        else:
            avgGain = avgGain * (rsiDays-1)/rsiDays
            avgLoss = ((avgLoss * (rsiDays-1)) + abs(diff))/rsiDays
            final.at[indx, 'RSI'] = 100.0 - (100/(1+(avgGain/avgLoss)))
#plt.subplot(211)
#plt.plot(final['Date'], final['Close'])
#plt.legend(['Close'])
#plt.grid(True, linestyle='--')
plt.subplot(212)
plt.plot(final['Date'], final['RSI'], 'k', final['Date'], [70] * final.shape[0], 'r--', final['Date'], [30] * final.shape[0], 'g--')
plt.legend(['RSI', 'Over sold', 'Over bought'])
plt.grid(True, linestyle='--')
plt.show()
