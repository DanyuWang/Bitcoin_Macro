import seaborn as sns
import matplotlib.pyplot as plt
from MA3.ML4Fin.Bitcoin_Macro.preprocess import *
from statsmodels.tsa.stattools import adfuller


daily, monthly = data_loader()


def get_correlation():
    month_average = daily.groupby(pd.PeriodIndex(daily.index, freq="M")).mean()
    month_date = monthly.groupby(pd.PeriodIndex(monthly.index, freq="M")).mean()

    all_month = pd.concat([month_average, month_date], axis=1)
    sns.heatmap(all_month.corr())
    plt.show()

    return all_month.corr()


def time_series_analysis(series):
    result = adfuller(series.dropna())
    if result[0] < result[4].get('5%'):
        print(series.name, "is stationary.")
    else:
        print(series.name, "is not stationary.")

    diff_s = series - series.shift()
    diff_res = adfuller(diff_s.dropna())
    if diff_res[0] < diff_res[4].get('5%'):
        print(series.name, "is 1-diff stationary.")
    else:
        print(series.name, "is not 1-diff stationary.")


def log_data(data, cols):
    for col in cols:
        data[col] = np.log(data[col])

    return data


def summary_analysis(dataframe):
    for col in dataframe.columns:
        time_series_analysis(dataframe[col])


if __name__ == '__main__':
    log_temp = log_data(daily, daily.columns)
    summary_analysis(log_temp)