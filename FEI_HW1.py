# ==========================================================================
# FINANCIAL ECONOMETRICS I
# Homework 1 Characteristics of Financial Time Series
#
# Authors : Maxime Borel, Florian Perusset & Keyu Zang 
# Date: OCT 2022
# ==========================================================================

import pandas as pd
import numpy as np 
import HW_fun as f
import matplotlib.pyplot as plt
from scipy import stats

# Q2
df = pd.read_excel( 'DATA_HW1.xlsx', header=1, index_col=0 )
df = df.drop( 'CRBSPOT(PI)', axis=1 )
df.index = pd.DatetimeIndex( df.index )
df.index.name = 'DATE'

simple_ret = df.pct_change().dropna()
log_ret = ( np.log( df.astype( 'float64' ) ) ).diff( 1 ).dropna()

n_asset = simple_ret.shape[ 1 ]

simple_daily_description = f.get_description_table( simple_ret )
simple_daily_description = simple_daily_description.round( 4 )
simple_daily_description.to_latex( 'results/simple_daily_description.tex' )

log_daily_description = f.get_description_table( log_ret )
log_daily_description = log_daily_description.round( 4 )
log_daily_description.to_latex( 'results/log_daily_description.tex' )

# Q2.a
f.get_fig_comp_ret( n_asset, simple_ret, log_ret )

g = plt.figure()
plt.plot( ( simple_ret - log_ret ), linewidth=0.5 )
plt.title( 'Daily difference' )
plt.xlabel( 'Time' )
plt.ylabel( 'Simple returns - log returns' )
plt.title( 'Difference between simple and log return' )
plt.legend( simple_ret.columns.to_list() )
g.savefig( 'results/diffretdailySimplevsLog', dpi=200 )

# Q2.b
# compute weekly returns
gross_ret = ( 1+simple_ret ).reset_index()
w_s_ret = gross_ret.resample( '1W', on='DATE' ).prod() - 1
w_s_ret = w_s_ret.drop( w_s_ret.index[ -1 ], axis=0 ) 
w_s_ret.index = simple_ret.index[ 4::5 ]

w_l_ret = np.log( 1+w_s_ret )

s_w_description = f.get_description_table( w_s_ret, ann_factor=52 ).round( 4 )
s_w_description.to_latex( 'results/s_w_description.tex' )

l_w_description = f.get_description_table( w_l_ret, ann_factor=52 ).round( 4 )
l_w_description.to_latex( 'results/l_w_description.tex' )

f.get_fig_comp_ret( n_asset, w_s_ret, w_l_ret, freq='w' )
g = plt.figure()
plt.plot( ( w_s_ret - w_l_ret ), linewidth=0.5 )
plt.title( 'Weekly difference' )
plt.xlabel( 'Time' )
plt.ylabel( 'Weekly Simple returns - log returns' )
plt.title( 'Difference between simple and log return' )
plt.legend( simple_ret.columns.to_list() )
g.savefig( 'results/diffretweeklySimplevsLog', dpi=200 )

# Q3
sp500_d = log_ret[ 'S&PCOMP(RI)' ]
sp500_w = w_l_ret[ 'S&PCOMP(RI)' ]

# Q3.a
nlargest_d = sp500_d.nlargest( 5 ).round( 4 )
nlargest_d.to_latex( 'results/nlargest_d.tex' )
nlargest_w = sp500_w.nlargest( 5 ).round( 4 )
nlargest_w.to_latex( 'results/nlargest_w.tex' )

nsmallest_d = sp500_d.nsmallest( 5 ).round( 4 )
nsmallest_d.to_latex( 'results/nsmallest_d.tex' )
nsmallest_w = sp500_w.nsmallest( 5 ).round( 4 )
nsmallest_w.to_latex( 'results/nsmallest_w.tex' )

# Q3.b
proba_crash_d, proba_boom_d = f.get_proba_of_extremes( log_ret )
proba_crash_w, proba_boom_w = f.get_proba_of_extremes( w_l_ret )

proba_crash_d.round( 4 ).to_latex( 'results/proba_crash_d.tex' )
proba_boom_d.round( 4 ).to_latex( 'results/proba_boom_d.tex' )
proba_crash_w.round( 4 ).to_latex( 'results/proba_crash_w.tex' )
proba_boom_w.round( 4 ).to_latex( 'results/proba_boom_w.tex' )

# Q3.c
jb_d = f.jarque_bera( log_ret ).round( 4 )
jb_w = f.jarque_bera( w_l_ret ).round( 4 )

jb_d.to_latex( 'results/jb_d.tex' )
jb_w.to_latex( 'results/jb_w.tex' )

# Q3.d
f.autocorr( log_ret )
f.autocorr( w_l_ret, freq='w' )

f.autocorr( log_ret**2, squared=True )
f.autocorr( w_l_ret**2, freq='w', squared=True )

d_ljb = f.ljungbox( log_ret, 10 )
w_ljb = f.ljungbox( w_l_ret, 10 )


r_d = pd.DataFrame()
for test, name in zip( d_ljb, simple_ret.columns.to_list() ):
    r_d[ name ] = test[ 'lb_stat' ]
r_d[ 'Critical Value' ] = pd.DataFrame([ stats.chi2.ppf( 0.95, i) for i in range(1, 11)], index=list(np.arange(1,11)))
r_d.to_latex( 'restults/ljb_daily.tex' )

r_w = pd.DataFrame()
for test, name in zip( w_ljb, simple_ret.columns.to_list() ):
    r_w[ name ] = test[ 'lb_stat' ]
r_w[ 'Critical Value' ] = r_d[ 'Critical Value' ] 
r_w.to_latex( 'restults/ljb_weekly.tex' )


d_ljb2 = f.ljungbox( log_ret**2, 10 )
w_ljb2 = f.ljungbox( w_l_ret**2, 10 )


r_d2 = pd.DataFrame()
for test, name in zip( d_ljb2, simple_ret.columns.to_list() ):
    r_d2[ name ] = test[ 'lb_stat' ]
r_d2[ 'Critical Value' ] = r_d[ 'Critical Value' ] 
r_d2.to_latex( 'restults/ljb_daily_squared.tex' )

r_w2 = pd.DataFrame()
for test, name in zip( w_ljb2, simple_ret.columns.to_list() ):
    r_w[ name ] = test[ 'lb_stat' ]
r_w2[ 'Critical Value' ] = r_d[ 'Critical Value' ] 
r_w2.to_latex( 'restults/ljb_weekly_squared.tex' )

# Q4
# Q4.a
d_port_ret = simple_ret.mean( 1 )
d_port_ret.columns = [ 'EW portfolio' ]
describe_d_port = f.get_description_table( d_port_ret ).round( 4 )
describe_d_port.to_latex( 'results/describe_d_port.tex' )

# Q4.b
w_port_ret = w_s_ret.mean( 1 )
describe_w_port = f.get_description_table( w_port_ret ).round( 4 )
describe_w_port.to_latex( 'results/describe_w_port.tex' )

g = plt.figure()
plt.plot( ( 1+d_port_ret ).cumprod(), linewidth=1 )
plt.plot( ( 1+simple_ret ).cumprod(), linewidth=1 )
plt.legend([ 'EW portfolio' ]+simple_ret.columns.to_list() )
plt.xlabel( 'Time' )
plt.ylabel( 'Cumulative product' )
g.savefig( 'results/portfolioAssets', dpi=200 )

