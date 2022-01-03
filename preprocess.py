import os
import numpy as np
import pandas as pd


DATAPATH = '/Users/holly/PycharmProjects/untitled/venv/MA3/ML4Fin/Bitcoin_Macro/data'


def read_csv(filename):
    if not os.path.exists(filename):
        print("The file of", filename, "doesn't exist.")

        return

    temp = pd.read_csv(filename, index_col=0)
    temp.index = pd.to_datetime(temp.index)

    return temp


def data_loader():
    # monthly data
    CPI = read_csv(os.path.join(DATAPATH, 'CPIAUCSL.csv'))
    M2 = read_csv(os.path.join(DATAPATH, 'M2SL.csv'))

    # daily data
    btc_usd = read_csv(os.path.join(DATAPATH, 'BTC-USD.csv'))
    flow_spl = read_csv(os.path.join(DATAPATH, 'BTC_flow-in_flow-out_supply_cex.csv'))
    exchange = average_flow_spl(flow_spl)
    FFR = read_csv(os.path.join(DATAPATH, 'DFF.csv'))
    gold = read_csv(os.path.join(DATAPATH, 'GOLDPMGBD228NLBM.csv'))

    # concat series
    daily = exchange.merge(btc_usd['Close'], left_index=True, right_index=True)
    daily = daily.merge(FFR,left_index=True, right_index=True)
    daily = pd.concat([daily, gold], axis=1)
    daily.columns = np.append(daily.columns[:3], ['btc_price', 'ffr', 'gold'])
    monthly = pd.concat([M2, CPI], axis=1)
    monthly.columns = ['M2', 'CPI']

    return daily, monthly


def average_flow_spl(data):
    flowin = data.loc[:, data.columns.str.contains('flow.in')]
    flowout = data.loc[:, data.columns.str.contains('flow.out')]
    supply = data.loc[:, data.columns.str.contains('sply')]

    in_wozero = flowin.replace(0.0, np.nan)
    out_wozero = flowout.replace(0.0, np.nan)
    supply_wozero = supply.replace(0.0, np.nan)

    flowin_mu = in_wozero.mean(axis=1)
    flowout_mu = out_wozero.mean(axis=1)
    supply_mu = supply_wozero.mean(axis=1)

    exchange = pd.concat([flowin_mu, flowout_mu, supply_mu], axis=1)
    exchange.columns = ['flow_in', 'flow_out', 'supply']
    exchange.index = exchange.index.tz_localize(None)

    return exchange


if __name__ == '__main__':
    daily, monthly = data_loader()
    print(daily.head())
    print(monthly.tail())