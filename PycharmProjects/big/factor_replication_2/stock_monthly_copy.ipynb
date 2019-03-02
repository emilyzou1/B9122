#!/usr/bin/env /share/share1/share_dff/anaconda3/bin/python

"""
Author: Lira Mota, lmota20@gsb.columbia.edu
Course: Big Data in Finance (Spring 2019)
Date: 2019-02
Code:
    Creates stock_monthly pandas data frame.
    Import CRSP MSE and MSF.

------

Dependence:
fire_pytools

"""

# %% Packages


import sys
sys.path.append('/Users/manshizou/PycharmProjects/big/')
import fire_pytools

import wrds
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import MonthEnd

from fire_pytools.import_wrds.crsp_sf import *
from fire_pytools.utils.post_event_nan import *


def calculate_melag(mdata):
    """
     Parameters:
    ------------
    mdata: data frame
        crsp monthly data with cols permno, date as index and lag_me column

    Notes:
    ------
    If ME is missing, we do not exclude stock, but rather keep it in with last non-missing MElag.
    The stock will be excluded if:
    (i) Delisted;
    (ii) Have a missing ME in the moment of portfolio construction.

    This is different than Ken's method

    EXAMPLE:
    --------
    there seem to be 12 stocks with missing PRC and thus missing ME in July 1926.
    Thus, our number of firms drops from 428 to 416.
    Fama and French report 427 in both July and August, so they also do not seem to exclude these
    rather they probably use the previous MElag for weight and must assume some return in the following month.

    The whole paragraph from the note on Ken French's website:
    ----------------------------------------------------------
    "In May 2015, we revised the method for computing daily portfolio returns
    to match more closely the method for computing monthly portfolio returns.
    Daily files produced before May 2015 drop stocks from a portfolio
    (i) the next time the portfolio is reconstituted, at the end of June, regardless of the CRSP delist date or
    (ii) during any period in which they are missing prices for more than 10 consecutive trading days.
    Daily files produced after May 2015 drop stocks from a portfolio
    (i) immediately after their CRSP delist date or
    (ii) during any period in which they are missing prices for more than 200 consecutive trading days. "
    """

    required_cols = ['lag_me', 'lag_dlret']

    set(required_cols).issubset(mdata.columns), "Required columns: {}.".format(', '.join(required_cols))

    df = mdata[required_cols].copy()
    df['melag'] = df.groupby('permno').lag_me.fillna(method='pad')
    df.reset_index(inplace=True)

    # Fill na after delisting
    df = post_event_nan(df=df, event=df.lag_dlret.notnull(), vars=['melag'], id_vars=['permno', 'edate'])

    df.set_index(['permno', 'edate'], inplace=True)

    return df[['melag']]

def calculate_cumulative_returns(mdata, tt, mp): #TODO: to be completed
    """
    Calculate past returns for momentum stratagy

    Parameters:
    ------------
    mdata: data frame
        crsp monthly data with cols permno, date as index.
    tt: int
        number of periods to cumulate retuns
    min_periods: int
        minimum number of periods. Default tt/2
    """
    start_time = time.time()

    required_cols = ['retadj']

    assert set(required_cols).issubset(mdata.columns), "Required columns: {}.".format(', '.join(required_cols))

    df = mdata[required_cols].copy()
    df['retadj'] = df['retadj']+1
    df['ret'] = df['retadj'].isnull()

    df.reset_index(level=0, inplace=True)

    cret = df.groupby('permno')['retadj'].rolling(window=tt, min_periods=mp).apply(np.nanprod, raw=True)
    cret_roll_back = df.groupby('permno')['retadj'].rolling(window=2, min_periods=1).apply(np.nanprod, raw=True)
    cret_fin = cret/cret_roll_back
    cret_fin = cret_fin-1
    cret = cret_fin.to_frame('ret' + str(11_1))
    #unique = len(df['permno'].unique())
   
    cret_copy = cret.copy()
    cret_copy.reset_index(inplace=True)
    
    unique_permno=pd.DataFrame(cret_copy.groupby(cret_copy.edate.dt.year)['permno'].unique())
    unique_no = unique_permno['permno'].apply(lambda x: len(x))
    
    cret_copy['isnull'] = cret_copy['ret111'].isnull()
    missing = cret_copy.groupby(cret_copy.edate.dt.year)['isnull'].sum()
    max_per_year = cret_copy.groupby(cret_copy.edate.dt.year)['ret111'].max()
    min_per_year = cret_copy.groupby(cret_copy.edate.dt.year)['ret111'].min()
    average_per_year = cret_copy.groupby(cret_copy.edate.dt.year)['ret111'].mean()
    print("Time to calculate %d months past returns: %s seconds" % (tt, str(round(time.time() - start_time, 2))))

    return cret,missing,unique_no,max_per_year, min_per_year, average_per_year




#main function





def main(save_out=True):

    # %% Set Up
    db = wrds.Connection(wrds_username='mzou20')  # make sure to configure wrds connector before hand.
    DATAPATH = "/Users/manshizou/Documents/Computingforbusiness/hw4_output_1" # where to save output?


    start_time = time.time()

    # %% Download CRSP data
    varlist = ['dlret', 'dlretx', 'exchcd', 'naics', 'permco', 'prc', 'ret', 'shrcd', 'shrout', 'siccd', 'ticker']

    start_date = '1925-01-01'  # '2017-01-01' #
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    freq = 'monthly'  # 'daily'
    permno_list =  None #[10001, 14593, 10107]
    shrcd_list = [10, 11]
    exchcd_list = [1, 2, 3]
    crspm = crsp_sf(varlist,
                    start_date,
                    end_date,
                    freq=freq,
                    permno_list=permno_list,
                    shrcd_list=shrcd_list,
                    exchcd_list=exchcd_list,
                    db=db)
    
    query = "SELECT caldt as date, t30ret as rf FROM crspq.mcti"
    rf = db.raw_sql(query, date_cols=['date'])
    del query

    # %% Create variables

    # Rankyear
    # Rankyear is the year where we ranked the stock, e.g., for the return of a stock in January 2001,
    # rankyear is 2000, because we ranked it in June 2000
    crspm['rankyear'] = crspm.date.dt.year
    crspm.loc[crspm.date.dt.month <= 6, 'rankyear'] = crspm.loc[crspm.date.dt.month <= 6, 'rankyear'] - 1

    # Returns adjusted for delisting
    crspm['retadj'] = ((1 + crspm['ret'].fillna(0)) * (1 + crspm['dlret'].fillna(0)) - 1)
    crspm.loc[crspm[['ret', 'dlret']].isnull().all(axis=1), 'retadj'] = np.nan

    # Create Market Equity (ME)
    # SHROUT is the number of publicly held shares, recorded in thousands. ME will be reported in 1,000,000 ($10^6$).
    # If the stock is delisted, we set ME to NaN.
    # Also, some companies have multiple shareclasses (=PERMNOs).
    # To get the company ME, we need to calculate the sum of ME over all shareclasses for one company (=PERMCO).
    # This is used for sorting, but not for weights.
    crspm['me'] = abs(crspm['prc']) * (crspm['shrout'] / 1000)

    # Create MEsum
    crspm['mesum_permco'] = crspm.groupby(['date', 'permco']).me.transform(np.sum, min_count=1)

    # Adjust for delisting
    crspm.loc[crspm.dlret.notnull(), 'me'] = np.nan
    crspm.loc[crspm.dlret.notnull(), 'mesum'] = np.nan

    # Resample data (This takes about 9 min)
    # CRSP data has skipping months.
    # Create line to missing  months to facilitate the calculation of lag/past returns
    start_time1 = time.time()
    crspm['edate'] = crspm['date'] + MonthEnd(0)
    crspm.sort_values(['permno', 'edate'], inplace=True)
    pk_integrity(crspm, ['permno', 'edate'])
    crspm.set_index('edate', inplace=True)

    # Resample to take care of missing months
    scrspm = crspm[['permno', 'me', 'dlret']].groupby('permno').resample('M').mean().drop(columns='permno')  # mean maintains nan
    scrspm = scrspm.groupby('permno').shift(1)
    scrspm.columns = ['lag_' + i for i in scrspm.columns]

    crspm.reset_index(inplace=True)
    crspm.set_index(['permno', 'edate'], inplace=True)

    crspm = crspm.join(scrspm, how='outer')

    print("Time to resample data: %s seconds" % str(time.time() - start_time1))

    # Create MElag
    crspm['melag'] = calculate_melag(crspm)

    # TODO: Calculate past 11, 1 returns
    cum_return,missing,unique_no,max_per_year,min_per_year,average_per_year = calculate_cumulative_returns(crspm, 13, 6)
    crspm = crspm.join(cum_return)

    # Delete rows that were not in the original data set
    crspm.dropna(subset=['date'], inplace=True)
    crspm.drop(columns=[x for x in crspm.columns if 'lag_' in x], inplace=True)

    crspm.sort_values(['permno', 'date'], inplace=True)

    print("Time to create CRSP monthly: %s seconds" % str(time.time() - start_time))

    if save_out:
        crspm.to_pickle(DATAPATH+'stock_monthly.pkl')
        print("Successfully saved stock_monthly.")
    return crspm,missing,unique_no,max_per_year, min_per_year,average_per_year


# %% Main
if __name__ == '__main__':
    crspm = main()





