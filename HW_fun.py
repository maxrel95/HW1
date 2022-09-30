import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import time 


freq_name = {
    'd': 'Daily',
    'w': 'Weekly'
}

def get_description_table( data, ann_factor=252 ):
    stats_name = [ 'Ann. mean', 'Ann. var', 'skew', 'kurt', 'min', 'max' ]
    res = pd.DataFrame(
        [
            data.mean()*ann_factor,
            data.var()*ann_factor,
            data.skew( axis=0 ),
            data.kurtosis( axis=0 ) + 3,
            data.min(),
            data.max()
        ],
        index=stats_name
    )
    return res

def get_fig_comp_ret( n_assets, simple, log, freq='d' ):
    for i in range( n_assets ):
        p = plt.figure()
        plt.plot( simple.iloc[ :, i ], 'k', linewidth=2 )
        plt.plot( log.iloc[ :, i ], 'r.', markersize=2 )
        plt.title( freq_name.get( freq )+' simple vs Log return for asset '+simple.columns.to_list()[ i ])
        plt.xlabel( 'Time' )
        plt.ylabel( 'Return' )
        plt.legend( [ 'Simple', 'Log' ] )
        p.savefig( 'results/'+freq+'_'+str( i ), dpi=200 )
        time.sleep( 0.5 )

def get_proba_of_extremes( data ):
    p = []
    p2 = []

    for asset in data.columns.to_list():
        serie = data[ asset ]
        nlargest = serie.nlargest( 5 )
        nsmallest = serie.nsmallest( 5 )

        mu = serie.mean()
        std = serie.std()

        p.append( pd.DataFrame( norm.cdf( nsmallest, loc=mu, scale=std ), columns=[ asset ] ) )
        p2.append( pd.DataFrame( 1 - norm.cdf( nlargest, loc=mu, scale=std ), columns=[ asset ] ) ) 

    res_crash = pd.concat( p, axis=1 )
    res_boom = pd.concat( p2, axis=1 )
    return res_crash, res_boom

def jarque_bera( data, freq='daily' ):
    res = []
    T = data.shape[ 0 ]

    for asset in data.columns.to_list():
        serie = data[ asset ]
        skew = stats.skew( serie )
        kurt = stats.kurtosis( serie )

        tstat = T*( ( skew**2 )/6 + ( kurt**2 )/24 )
        res.append( tstat )
    
    pval = stats.chi2.ppf( 0.95, 2 )

    return pd.DataFrame( [res, pval], columns=data.columns.to_list(), index=['JBTest '+freq, 'pvalue'] )


def var_rho( rho, T ):
    v = []
    for k in range( 1, rho.shape[ 0 ]+1 ):
        t = T - k
        var = ( 1/t )*(1 + 2*( np.sum( rho[ :k ]**2 ) ) )
        v.append( var )
    return v

def autocorr( data, lag=10, freq='d', squared=False ):
    T = data.iloc[:, 0].shape[ 0 ]
    for asset in data.columns.to_list():
        serie = data[ asset ]
        ac = acf( serie, nlags=lag )
        v = var_rho( ac, T )
        ci = 1.96*np.sqrt( v[ :-1 ] )

        if squared:
            name = 'squared'
        else:
            name = ''

        g = plt.figure()
        plt.plot(ac[ 1: ], 'k')
        plt.plot(ci, 'b--')
        plt.plot(-ci, 'b--')
        plt.ylim([ -np.max( np.abs( ac[ 1: ] ) )-0.1, np.max( np.abs( ac[ 1: ] ) )+0.1])
        plt.xlabel( 'lag' )
        plt.ylabel( 'autocorrelation' )
        plt.title( freq_name.get( freq )+' autocorrelation for '+name+' return of '+asset )
        g.savefig( 'results/autocorrelation_'+name+'_'+asset+'_'+freq, dpi=200 )
        time.sleep( 0.5 )



def ljungbox( data, lag ):
    res = []
    for asset in data.columns.to_list():
        serie = data[ asset ]
        r = acorr_ljungbox( serie, lags=lag, return_df=True)
        res.append( r )

    return res
