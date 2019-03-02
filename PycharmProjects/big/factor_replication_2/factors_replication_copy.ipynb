"""
Author: Lira Mota
Date: 2019-02
Code: Replicates Fama and French 5 Factor (work in progress)
"""

#%% Packages

import sys
sys.path.append('/Users/manshizou/PycharmProjects/big/')
import fire_pytools

import pandas as pd
from pylab import *
from matplotlib.pyplot import figure

import stock_annual as stock_annual
import stock_monthly as stock_monthly

from fire_pytools.portools.sort_portfolios import sort_portfolios

idx = pd.IndexSlice

# %% Set Up

char_breakpoints = {'mesum_dec': [0.5],
                    'beme': [0.3, 0.7],
                    'opbe': [0.3, 0.7],
                    'inv_gvkey': [0.3, 0.7]}

weightvar = 'melag_weight'

retvar = 'retadj'

rankvar = 'rankyear'

dict_factors = {'BEME': 'HML',
                'ME': 'SMB',
                'OP': 'RMW',
                'INV': 'CMA'}

# %% Download Data

# Annual Data
adata = stock_annual.main()

adata = adata.dropna(subset=['mesum_dec','mesum_june'], how = 'any')

# Monthly Data
mdata,missing,unique,max_per_year, min_per_year, average_per_year= stock_monthly.main()


# %% Find Breakpoints

adata_me = adata
me_bp =find_breakpoints(adata_me,quantiles={'mesum_dec': [0.5]},id_variables = ['rankyear', 'permno', 'exchcd'],exch_cd=[1], silent=False)


adata_beme = adata[adata.be > 0]
adata_beme = adata_beme.dropna(subset=['beme'],how='any')
beme_bp = find_breakpoints(adata_beme,quantiles={'beme': [0.3,0.7]} ,id_variables = ['rankyear', 'permno', 'exchcd'],exch_cd=[1], silent=False)

adata_opbe = adata[adata.be > 0]
adata_opbe = adata_opbe.dropna(subset=['opbe'],how='any')
opbe_bp = find_breakpoints(adata_opbe,quantiles={'opbe': [0.3,0.7]} ,id_variables = ['rankyear', 'permno', 'exchcd'],exch_cd=[1], silent=False)



adata_inv = adata.dropna(subset=['inv_gvkey'], how='any' )
inv_bp = find_breakpoints(adata_inv,quantiles={'inv_gvkey': [0.3,0.7]} ,id_variables = ['rankyear', 'permno', 'exchcd'],exch_cd=[1], silent=False)

mdata_mom = mdata.dropna(subset=['ret111'],how='any')
mom_bp = find_breakpoints(mdata_mom,quantiles={'ret111': [0.3,0.7]} ,id_variables = ['date', 'permno', 'exchcd'],exch_cd=[1], silent=False)

mdata_me = mdata.dropna(subset=['melag'],how='any')
mem_bp = find_breakpoints(mdata_me,quantiles={'melag': [0.5]} ,id_variables = ['date', 'permno', 'exchcd'],exch_cd=[1], silent=False)



# %% Portfolio Sorts
port_me = sort_portfolios(adata_me, quantiles = {'mesum_dec': [0.5]},id_variables = ['rankyear', 'permno', 'exchcd'], breakpoints=me_bp, exch_cd=None, silent=False, numeric_ports=False)

port_beme = sort_portfolios(adata_beme, quantiles = {'beme': [0.3,0.7]},id_variables = ['rankyear', 'permno', 'exchcd'], breakpoints=beme_bp, exch_cd=None, silent=False, numeric_ports=False)

port_opbe = sort_portfolios(adata_opbe, quantiles = {'opbe': [0.3,0.7]}, id_variables = ['rankyear', 'permno', 'exchcd'], breakpoints=opbe_bp, exch_cd=None, silent=False, numeric_ports=False)

port_inv = sort_portfolios(adata_inv, quantiles ={'inv_gvkey': [0.3,0.7]} , id_variables = ['rankyear', 'permno', 'exchcd'], breakpoints=inv_bp, exch_cd=None, silent=False, numeric_ports=False)

port_mom = sort_portfolios(mdata_mom, quantiles ={'ret111': [0.3,0.7]} , id_variables = ['date', 'permno', 'exchcd'], breakpoints=mom_bp, exch_cd=None, silent=False, numeric_ports=False)

port_mem = sort_portfolios(mdata_me, quantiles ={'melag': [0.5]} , id_variables = ['date', 'permno', 'exchcd'], breakpoints=mem_bp, exch_cd=None, silent=False, numeric_ports=False)

#unique_co = port.groupby('rankyear')['permno'].unique()
# %% Portfolio Returns
f_beme = pd.merge(mdata, port_beme,  how='inner', left_on=['rankyear','permno'], right_on = ['rankyear','permno'])

f_opbe = pd.merge(f_beme, port_opbe,  how='inner', left_on=['rankyear','permno'], right_on = ['rankyear','permno'])

f_me = pd.merge(f_opbe, port_me,  how='inner', left_on=['rankyear','permno'], right_on = ['rankyear','permno'])

f = pd.merge(f_me, port_inv, how='inner', left_on=['rankyear','permno'], right_on = ['rankyear','permno'])


##You use mesum to sort and use ME as weights
##


f = f.sort_values('date')
f.set_index('date',inplace = True)
f = f.dropna(subset =['retadj','melag'])
#Calculate beme3 me1 averaged weighted return
sum_beme3_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['bemeportfolio']=='beme3')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_beme3_me1 = pd.DataFrame(f.groupby('date', as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme3')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_beme3_me1 = pd.merge(sum_beme3_me1,p_beme3_me1, how='left',left_on = ['date'],right_on =['date'])
weight_beme3_me1 = pd.DataFrame(np.divide(pp_beme3_me1['melag'],pp_beme3_me1[0]))
ret_beme3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme3')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_beme3_me1.index = ret_beme3_me1.index.droplevel(level=0)
weighted_ret_beme3_me1 =ret_beme3_me1['retadj']*weight_beme3_me1['melag']
beme3_me1 = weighted_ret_beme3_me1.groupby('date').sum()
beme3_me1 = beme3_me1.sort_index()

#Calculate beme3 me2 averaged weighted return

sum_beme3_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['bemeportfolio']=='beme3')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_beme3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme3')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_beme3_me2 = pd.merge(sum_beme3_me2,p_beme3_me2, how='left',left_on = ['date'],right_on =['date'])
weight_beme3_me2 = pd.DataFrame(np.divide(pp_beme3_me2['melag'],pp_beme3_me2[0]))
ret_beme3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme3')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_beme3_me2.index = ret_beme3_me2.index.droplevel(level=0)
weighted_ret_beme3_me2 =ret_beme3_me2['retadj']*weight_beme3_me2['melag']
beme3_me2 = weighted_ret_beme3_me2.groupby('date').sum()

#Calculate beme2 me1 averaged weighted return

sum_beme2_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['bemeportfolio']=='beme2')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_beme2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme2')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_beme2_me1 = pd.merge(sum_beme2_me1,p_beme2_me1, how='left',left_on = ['date'],right_on =['date'])
weight_beme2_me1 = pd.DataFrame(np.divide(pp_beme2_me1['melag'],pp_beme2_me1[0]))
ret_beme2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme2')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_beme2_me1.index = ret_beme2_me1.index.droplevel(level=0)
weighted_ret_beme2_me1 =ret_beme2_me1['retadj']*weight_beme2_me1['melag']
beme2_me1 = weighted_ret_beme2_me1.groupby('date').sum()

#Calculate beme2 me2 averaged weighted return
sum_beme2_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['bemeportfolio']=='beme2')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_beme2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme2')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_beme2_me2 = pd.merge(sum_beme2_me2,p_beme2_me2, how='left',left_on = ['date'],right_on =['date'])
weight_beme2_me2 = pd.DataFrame(np.divide(pp_beme2_me2['melag'],pp_beme2_me2[0]))
ret_beme2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme2')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_beme2_me2.index = ret_beme2_me2.index.droplevel(level=0)
weighted_ret_beme2_me2 =ret_beme2_me2['retadj']*weight_beme2_me2['melag']
beme2_me2 = weighted_ret_beme2_me2.groupby('date').sum()

#Calculate beme1 me1 averaged weighted return
sum_beme1_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['bemeportfolio']=='beme1')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_beme1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme1')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_beme1_me1 = pd.merge(sum_beme1_me1,p_beme1_me1, how='left',left_on = ['date'],right_on =['date'])
weight_beme1_me1 = pd.DataFrame(np.divide(pp_beme1_me1['melag'],pp_beme1_me1[0]))
ret_beme1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme1')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_beme1_me1.index = ret_beme1_me1.index.droplevel(level=0)
weighted_ret_beme1_me1 =ret_beme1_me1['retadj']*weight_beme1_me1['melag']
beme1_me1 = weighted_ret_beme1_me1.groupby('date').sum()

#Calculate beme1 me2 averaged weighted return
sum_beme1_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['bemeportfolio']=='beme1')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_beme1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme1')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_beme1_me2 = pd.merge(sum_beme1_me2,p_beme1_me2, how='left',left_on = ['date'],right_on =['date'])
weight_beme1_me2 = pd.DataFrame(np.divide(pp_beme1_me2['melag'],pp_beme1_me2[0]))
ret_beme1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['bemeportfolio']=='beme1')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_beme1_me2.index = ret_beme1_me2.index.droplevel(level=0)
weighted_ret_beme1_me2 =ret_beme1_me2['retadj']*weight_beme1_me2['melag']
beme1_me2 = weighted_ret_beme1_me2.groupby('date').sum()


smb_comp1 = (beme3_me1+beme2_me1+beme1_me1-beme3_me2-beme2_me2-beme1_me2)/3

###OP
#Calculate op3 me1 averaged weighted return
sum_op3_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['opbeportfolio']=='opbe3')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_op3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe3')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_op3_me1 = pd.merge(sum_op3_me1,p_op3_me1, how='left',left_on = ['date'],right_on =['date'])
weight_op3_me1 = pd.DataFrame(np.divide(pp_op3_me1['melag'],pp_op3_me1[0]))
ret_op3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe3')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_op3_me1.index = ret_op3_me1.index.droplevel(level=0)
weighted_ret_op3_me1 =ret_op3_me1['retadj']*weight_op3_me1['melag']
op3_me1 = weighted_ret_op3_me1.groupby('date').sum()

#Calculate op3 me2 averaged weighted return

sum_op3_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['opbeportfolio']=='opbe3')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_op3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe3')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_op3_me2 = pd.merge(sum_op3_me2,p_op3_me2, how='left',left_on = ['date'],right_on =['date'])
weight_op3_me2 = pd.DataFrame(np.divide(pp_op3_me2['melag'],pp_op3_me2[0]))
ret_op3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe3')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_op3_me2.index = ret_op3_me2.index.droplevel(level=0)
weighted_ret_op3_me2 =ret_op3_me2['retadj']*weight_op3_me2['melag']
op3_me2 = weighted_ret_op3_me2.groupby('date').sum()

#Calculate op2 me1 averaged weighted return

sum_op2_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['opbeportfolio']=='opbe2')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_op2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe2')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_op2_me1 = pd.merge(sum_op2_me1,p_op2_me1, how='left',left_on = ['date'],right_on =['date'])
weight_op2_me1 = pd.DataFrame(np.divide(pp_op2_me1['melag'],pp_op2_me1[0]))
ret_op2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe2')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_op2_me1.index = ret_op2_me1.index.droplevel(level=0)
weighted_ret_op2_me1 =ret_op2_me1['retadj']*weight_op2_me1['melag']
op2_me1 = weighted_ret_op2_me1.groupby('date').sum()

#Calculate op2 me2 averaged weighted return
sum_op2_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['opbeportfolio']=='opbe2')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_op2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe2')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_op2_me2 = pd.merge(sum_op2_me2,p_op2_me2, how='left',left_on = ['date'],right_on =['date'])
weight_op2_me2 = pd.DataFrame(np.divide(pp_op2_me2['melag'],pp_op2_me2[0]))
ret_op2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe2')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_op2_me2.index = ret_op2_me2.index.droplevel(level=0)
weighted_ret_op2_me2 =ret_op2_me2['retadj']*weight_op2_me2['melag']
op2_me2 = weighted_ret_op2_me2.groupby('date').sum()

#Calculate op1 me1 averaged weighted return
sum_op1_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['opbeportfolio']=='opbe1')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_op1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe1')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_op1_me1 = pd.merge(sum_op1_me1,p_op1_me1, how='left',left_on = ['date'],right_on =['date'])
weight_op1_me1 = pd.DataFrame(np.divide(pp_op1_me1['melag'],pp_op1_me1[0]))
ret_op1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe1')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_op1_me1.index = ret_op1_me1.index.droplevel(level=0)
weighted_ret_op1_me1 =ret_op1_me1['retadj']*weight_op1_me1['melag']
op1_me1 = weighted_ret_op1_me1.groupby('date').sum()

#Calculate op1 me2 averaged weighted return
sum_op1_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['opbeportfolio']=='opbe1')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_op1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe1')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_op1_me2 = pd.merge(sum_op1_me2,p_op1_me2, how='left',left_on = ['date'],right_on =['date'])
weight_op1_me2 = pd.DataFrame(np.divide(pp_op1_me2['melag'],pp_op1_me2[0]))
ret_op1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['opbeportfolio']=='opbe1')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_op1_me2.index = ret_op1_me2.index.droplevel(level=0)
weighted_ret_op1_me2 =ret_op1_me2['retadj']*weight_op1_me2['melag']
op1_me2 = weighted_ret_op1_me2.groupby('date').sum()


smb_comp2 = (op3_me1+op2_me1+op1_me1-op3_me2-op2_me2-op1_me2)/3


###inv
#Calculate inv3 me1 averaged weighted return
sum_inv3_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey3')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_inv3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey3')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_inv3_me1 = pd.merge(sum_inv3_me1,p_inv3_me1, how='left',left_on = ['date'],right_on =['date'])
weight_inv3_me1 = pd.DataFrame(np.divide(pp_inv3_me1['melag'],pp_inv3_me1[0]))
ret_inv3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey3')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_inv3_me1.index = ret_inv3_me1.index.droplevel(level=0)
weighted_ret_inv3_me1 =ret_inv3_me1['retadj']*weight_inv3_me1['melag']
inv3_me1 = weighted_ret_inv3_me1.groupby('date').sum()

#Calculate inv3 me2 averaged weighted return

sum_inv3_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey3')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_inv3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey3')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_inv3_me2 = pd.merge(sum_inv3_me2,p_inv3_me2, how='left',left_on = ['date'],right_on =['date'])
weight_inv3_me2 = pd.DataFrame(np.divide(pp_inv3_me2['melag'],pp_inv3_me2[0]))
ret_inv3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey3')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_inv3_me2.index = ret_inv3_me2.index.droplevel(level=0)
weighted_ret_inv3_me2 =ret_inv3_me2['retadj']*weight_inv3_me2['melag']
inv3_me2 = weighted_ret_inv3_me2.groupby('date').sum()

#Calculate inv2 me1 averaged weighted return

sum_inv2_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey2')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_inv2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey2')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_inv2_me1 = pd.merge(sum_inv2_me1,p_inv2_me1, how='left',left_on = ['date'],right_on =['date'])
weight_inv2_me1 = pd.DataFrame(np.divide(pp_inv2_me1['melag'],pp_inv2_me1[0]))
ret_inv2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey2')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_inv2_me1.index = ret_inv2_me1.index.droplevel(level=0)
weighted_ret_inv2_me1 =ret_inv2_me1['retadj']*weight_inv2_me1['melag']
inv2_me1 = weighted_ret_inv2_me1.groupby('date').sum()

#Calculate inv2 me2 averaged weighted return
sum_inv2_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey2')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_inv2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey2')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_inv2_me2 = pd.merge(sum_inv2_me2,p_inv2_me2, how='left',left_on = ['date'],right_on =['date'])
weight_inv2_me2 = pd.DataFrame(np.divide(pp_inv2_me2['melag'],pp_inv2_me2[0]))
ret_inv2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey2')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_inv2_me2.index = ret_inv2_me2.index.droplevel(level=0)
weighted_ret_inv2_me2 =ret_inv2_me2['retadj']*weight_inv2_me2['melag']
inv2_me2 = weighted_ret_inv2_me2.groupby('date').sum()

#Calculate inv1 me1 averaged weighted return
sum_inv1_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey1')&(x['mesum_decportfolio']=='mesum_dec1')]['melag'].sum()))
p_inv1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey1')&(x['mesum_decportfolio']=='mesum_dec1')]['melag']))
pp_inv1_me1 = pd.merge(sum_inv1_me1,p_inv1_me1, how='left',left_on = ['date'],right_on =['date'])
weight_inv1_me1 = pd.DataFrame(np.divide(pp_inv1_me1['melag'],pp_inv1_me1[0]))
ret_inv1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey1')&(x['mesum_decportfolio']=='mesum_dec1')]['retadj']))
ret_inv1_me1.index = ret_inv1_me1.index.droplevel(level=0)
weighted_ret_inv1_me1 =ret_inv1_me1['retadj']*weight_inv1_me1['melag']
inv1_me1 = weighted_ret_inv1_me1.groupby('date').sum()

#Calculate inv1 me2 averaged weighted return
sum_inv1_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey1')&(x['mesum_decportfolio']=='mesum_dec2')]['melag'].sum()))
p_inv1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey1')&(x['mesum_decportfolio']=='mesum_dec2')]['melag']))
pp_inv1_me2 = pd.merge(sum_inv1_me2,p_inv1_me2, how='left',left_on = ['date'],right_on =['date'])
weight_inv1_me2 = pd.DataFrame(np.divide(pp_inv1_me2['melag'],pp_inv1_me2[0]))
ret_inv1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['inv_gvkeyportfolio']=='inv_gvkey1')&(x['mesum_decportfolio']=='mesum_dec2')]['retadj']))
ret_inv1_me2.index = ret_inv1_me2.index.droplevel(level=0)
weighted_ret_inv1_me2 =ret_inv1_me2['retadj']*weight_inv1_me2['melag']
inv1_me2 = weighted_ret_inv1_me2.groupby('date').sum()







smb_comp3 = (inv3_me1+inv2_me1+inv1_me1-inv3_me2-inv2_me2-inv1_me2)/3


smb = (smb_comp1+smb_comp2+smb_comp3)/3


hml = (beme3_me1+beme3_me2-beme1_me1-beme1_me2)/2

rmw = (op3_me1+op3_me2-op1_me1-op1_me2)/2

inv = (inv1_me1+inv1_me2-inv3_me1-inv3_me2)/2

del f
#MOM indicators
#char_breakpoints2 = {'ret111': [0.3,0.7],'melag': [0.5]}

f_mom = pd.merge(mdata,port_mom, how = 'inner', left_on=['date','permno'], right_on = ['date','permno'])
f = pd.merge(f_mom, port_mem, how = 'inner', left_on=['date','permno'], right_on = ['date','permno'] )

f = f.sort_values('date')
f.set_index('date',inplace = True)


#Calculate momentum3 me1 averaged weighted return

sum_m3_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['ret111portfolio']=='ret1113')&(x['melagportfolio']=='melag1')]['melag'].sum()))
p_m3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1113')&(x['melagportfolio']=='melag1')]['melag']))
pp_m3_me1 = pd.merge(sum_m3_me1,p_m3_me1, how='left',left_on = ['date'],right_on =['date'])
weight_m3_me1 = pd.DataFrame(np.divide(pp_m3_me1['melag'],pp_m3_me1[0]))
ret_m3_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1113')&(x['melagportfolio']=='melag1')]['retadj']))
ret_m3_me1.index = ret_m3_me1.index.droplevel(level=0)
weighted_ret_m3_me1 =ret_m3_me1['retadj']*weight_m3_me1['melag']
m3_me1 = weighted_ret_m3_me1.groupby('date').sum()

#Calculate momentum3 me2 averaged weighted return
sum_m3_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['ret111portfolio']=='ret1113')&(x['melagportfolio']=='melag2')]['melag'].sum()))
p_m3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1113')&(x['melagportfolio']=='melag2')]['melag']))
pp_m3_me2 = pd.merge(sum_m3_me2,p_m3_me2, how='left',left_on = ['date'],right_on =['date'])
weight_m3_me2 = pd.DataFrame(np.divide(pp_m3_me2['melag'],pp_m3_me2[0]))
ret_m3_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1113')&(x['melagportfolio']=='melag2')]['retadj']))
ret_m3_me2.index = ret_m3_me2.index.droplevel(level=0)
weighted_ret_m3_me2 =ret_m3_me2['retadj']*weight_m3_me2['melag']
m3_me2 = weighted_ret_m3_me2.groupby('date').sum()


#Calculate momentum2 me1 averaged weighted return

sum_m2_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['ret111portfolio']=='ret1112')&(x['melagportfolio']=='melag1')]['melag'].sum()))
p_m2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1112')&(x['melagportfolio']=='melag1')]['melag']))
pp_m2_me1 = pd.merge(sum_m2_me1,p_m2_me1, how='left',left_on = ['date'],right_on =['date'])
weight_m2_me1 = pd.DataFrame(np.divide(pp_m2_me1['melag'],pp_m2_me1[0]))
ret_m2_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1112')&(x['melagportfolio']=='melag1')]['retadj']))
ret_m2_me1.index = ret_m2_me1.index.droplevel(level=0)
weighted_ret_m2_me1 =ret_m2_me1['retadj']*weight_m2_me1['melag']
m2_me1 = weighted_ret_m2_me1.groupby('date').sum()

#Calculate momentum2 me2 averaged weighted return
sum_m2_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['ret111portfolio']=='ret1112')&(x['melagportfolio']=='melag2')]['melag'].sum()))
p_m2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1112')&(x['melagportfolio']=='melag2')]['melag']))
pp_m2_me2 = pd.merge(sum_m2_me2,p_m2_me2, how='left',left_on = ['date'],right_on =['date'])
weight_m2_me2 = pd.DataFrame(np.divide(pp_m2_me2['melag'],pp_m2_me2[0]))
ret_m2_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1112')&(x['melagportfolio']=='melag2')]['retadj']))
ret_m2_me2.index = ret_m2_me2.index.droplevel(level=0)
weighted_ret_m2_me2 =ret_m2_me2['retadj']*weight_m2_me2['melag']
m2_me2 = weighted_ret_m2_me2.groupby('date').sum()

#Calculate momentumen1 me1 averaged weighted return
sum_m1_me1 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['ret111portfolio']=='ret1111')&(x['melagportfolio']=='melag1')]['melag'].sum()))
p_m1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1111')&(x['melagportfolio']=='melag1')]['melag']))
pp_m1_me1 = pd.merge(sum_m1_me1,p_m1_me1, how='left',left_on = ['date'],right_on =['date'])
weight_m1_me1 = pd.DataFrame(np.divide(pp_m1_me1['melag'],pp_m1_me1[0]))
ret_m1_me1 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1111')&(x['melagportfolio']=='melag1')]['retadj']))
ret_m1_me1.index = ret_m1_me1.index.droplevel(level=0)
weighted_ret_m1_me1 =ret_m1_me1['retadj']*weight_m1_me1['melag']
m1_me1 = weighted_ret_m1_me1.groupby('date').sum()

#Calculate momentum1 me2 averaged weighted return
sum_m1_me2 = pd.DataFrame(f.groupby('date').apply(lambda x: x[(x['ret111portfolio']=='ret1111')&(x['melagportfolio']=='melag2')]['melag'].sum()))
p_m1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1111')&(x['melagportfolio']=='melag2')]['melag']))
pp_m1_me2 = pd.merge(sum_m1_me2,p_m1_me2, how='left',left_on = ['date'],right_on =['date'])
weight_m1_me2 = pd.DataFrame(np.divide(pp_m1_me2['melag'],pp_m1_me2[0]))
ret_m1_me2 = pd.DataFrame(f.groupby('date',as_index = False).apply(lambda x: x[(x['ret111portfolio']=='ret1111')&(x['melagportfolio']=='melag2')]['retadj']))
ret_m1_me2.index = ret_m1_me2.index.droplevel(level=0)
weighted_ret_m1_me2 =ret_m1_me2['retadj']*weight_m1_me2['melag']
m1_me2 = weighted_ret_m1_me2.groupby('date').sum()



MOM = (m3_me1+m3_me2-m1_me1-m1_me2)/2

query = "SELECT caldt as date, t30ret as rf FROM crspq.mcti"
import wrds
db = wrds.Connection(wrds_username='mzou20') 
rf = db.raw_sql(query, date_cols=['date'])

m_m = pd.DataFrame(f.groupby('date')['melag'].sum())
individual_m = pd.DataFrame(f['melag'])
weight_merged = pd.merge(m_m,individual_m,how='left',left_on = ['date'],right_on =['date'])
symbol = ['sum','me']
weight_merged.columns = symbol
weight_calculated = pd.DataFrame(np.divide(weight_merged['me'],weight_merged['sum']))

weighted_ret = f['retadj']*weight_calculated['me']
weighted_m_ret = pd.DataFrame(weighted_ret.groupby('date').sum())


m_rf = pd.merge(weighted_m_ret, rf, how ='left',left_on = ['date'],right_on =['date'])

m_rf.set_index('date',inplace = True)
Rmrf = np.subtract(m_rf[0],m_rf['rf'])

Total = pd.concat([Rmrf,smb,hml,rmw,inv,MOM],axis = 1)
Total = Total.dropna()
symbol = ['Rmrf','SMB','HML','RMW','CMA','MOM']
Total.columns = symbol
# %% Checks
plt.plot(smb)
((smb+1).cumprod()-1).plot()
plt.plot(hml)
((hml+1).cumprod()-1).plot()
((rmw+1).cumprod()-1).plot()
plt.plot(MOM)
((inv+1).cumprod()-1).plot()
((MOM+1).csumprod()-1).plot()
((Rmrf+1).cumprod()-1).plot()
plt.show()
from datetime import datetime
ff_data = pd.read_csv('/Users/manshizou/Documents/Computingforbusiness/F-F.csv')
mm_data =pd.read_csv('/Users/manshizou/Downloads/mom.csv')
symbol1 = ['Date','MOM']
mm_data.columns = symbol
symbol=['Date','Mkt-RF','SMB','HML','RMW','CMA','RF']
ff_data.columns = symbol
ff_data['Date']= ff_data['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m'))
ff_data['Date']=ff_data['Date']+MonthEnd(0)

mm_data['month']= mm_data['month'].apply(lambda x: datetime.strptime(str(x),'%Y%m'))
mm_data['month']=mm_data['month']+MonthEnd(0)
mm_data.set_index(mm_data['month'],inplace =True)


Final =pd.merge(Total, ff_data[:-1], how = 'inner', left_index = True, right_on=['Date'])
corr_table = Final.corr()
corr_table.iloc[0:6,6:]
Final =pd.merge(Total, mm_data, how = 'inner', left_on=['date'], right_on = ['month'])
corr_table = Final.corr()
corr_table.iloc[0:6,6:]
