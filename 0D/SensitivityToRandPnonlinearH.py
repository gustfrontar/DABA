# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:30:13 2019
@author: jruiz
"""


import numpy as np
import matplotlib.pyplot as plt
import common_function as cf
plt.rcParams.update({'font.size': 15})



np.random.seed(70)

#PARAMETERS ---------

ens_size = 200                       #Size of the ensemble     
x_mean   = 5.0                       #Mean of the original state variable.


x_std_range=np.arange(0.1,5,0.5)
R_range=np.arange(0.1,10,1.0)
True_range   = np.arange(0.0,15.0,1.0)
x_mean_range = np.arange(0.0,15.0,1.0)

#alpha = cf.get_temp_steps( Ntemp , 2.0 )
#alpha=1 / ( ( 1/alpha ) /  np.sum( 1/alpha ) )

error_reduction_kf = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
update_size_kf     = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )

error_reduction_pf = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
update_size_pf     = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )

zero_per           = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
prior_error        = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
post_error_kf      = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
post_error_pf      = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
post_sprd_kf       = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )
prior_sprd         = np.zeros( ( len( True_range ) , len(x_mean_range) , len( x_std_range ) , len( R_range ) ) )

#Define the true and the observations
for zz , x_mean in enumerate( x_mean_range ) :
    print(zz,len(x_mean_range))
    for ii , x_true in enumerate( True_range ) :
        
        dbz_true = cf.calc_ref_bis( np.array( x_true ) )
        dbz_obs = dbz_true #Assuming a perfect observation (so we discard the effect of obs error)
        #dbz_obs = dbz_true + (R**0.5) * np.random.randn( 1 )
    
           
        for jj , x_std in enumerate( x_std_range ) :
            #--------------------
            #Define x ensemble
            x_ens = x_mean+x_std*np.random.randn( ens_size )
    
            #First compute the ensemble in terms of graupel concentration.
            dbz_ens = np.zeros( np.shape( x_ens ) )   
            #For each graupel concentration compute the corresponding reflectivity using 
            #the non-linear observation operator. 
            dbz_ens = cf.calc_ref_bis( x_ens )
            
            for kk, R in enumerate( R_range ) : 
                dbz_a_ens , x_a_ens = cf.calc_etkf_filter_update( dbz_ens , x_ens , dbz_obs , R )
                pf_w = np.exp(-( dbz_ens - dbz_obs )**2/R  )
                pf_w = pf_w / np.sum( pf_w)
                dbz_a_ens[ dbz_a_ens < 0 ] = 0
        
                error_reduction_kf[zz,ii,jj,kk]  = np.abs( np.mean( x_a_ens ) - x_true ) / np.abs( np.mean( x_ens ) - x_true )
                error_reduction_pf[zz,ii,jj,kk]  = np.abs( np.sum( x_ens * pf_w ) - x_true ) / np.abs( np.mean( x_ens ) - x_true )
            
                update_size_kf[zz,ii,jj,kk] = np.abs( np.mean( x_a_ens ) - np.mean( x_ens ) )
                update_size_pf[zz,ii,jj,kk] = np.abs( np.sum( x_ens * pf_w ) - np.mean( x_ens ) )
                
                zero_per[zz,ii,jj,kk] = np.mean( dbz_ens == 0.0 )
                
                prior_error[zz,ii,jj,kk]   = np.mean( x_ens ) - x_true
                post_error_kf[zz,ii,jj,kk] = np.mean( x_a_ens ) - x_true
                post_error_pf[zz,ii,jj,kk] = np.sum( x_ens * pf_w ) - x_true
                
                post_sprd_kf[zz,ii,jj,kk] = np.std( x_a_ens )
                prior_sprd[zz,ii,jj,kk]   = np.std( x_ens )
            



