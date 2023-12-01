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


np.random.seed(15)

#PARAMETERS ---------

ens_size = 5000                      #Size of the ensemble     
qr_mean  = 1.0e-4                    #Mean of graupel concentration
qr_std   = 5.0e-5                    #Standard deviation of graupel concentration.
dbz_tr   = -20.0                     #Minimum dBZ value.
qr_model_bias = 0.0e-4               #When generating the true qr we can add a model bias.                           
R=np.power( 5.0 , 2 )                #Observation error for reflectivity

alpha = np.array([90 , 5,  1])      #Tempering coefficients are computed normalizing the inverse of the components of this vector.
                                     #the larger the value, the less observations information that is assimilated in each iteration.
                                     #Usually values are greater for the first iteration an smaller for the rest of them. 

alpha=1 / ( ( 1/alpha ) /  np.sum( 1/alpha ) )

#--------------------
#Define qr ensemble
    
    
qr_ens = qr_mean+qr_std*np.random.randn( ens_size )

qr_ens[ qr_ens < 0.0 ] = 0.0

#Define the true and the observations

qr_true = qr_mean + 0.8e-4 #qr_std*np.random.randn( 1 ) + qr_model_bias
if qr_true < 0.0 :
    qr_true = 0.0

dbz_true = cf.calc_ref( qr_true )
dbz_obs = dbz_true + (R**0.5) * np.random.randn( 1 )
if dbz_obs < dbz_tr :
    dbz_obs = dbz_tr

#First compute the ensemble in terms of graupel concentration.
dbz_ens = np.zeros( np.shape( qr_ens ) )   
#For each graupel concentration compute the corresponding reflectivity using 
#the non-linear observation operator.
for ii in range( ens_size ):
    dbz_ens[ii] = cf.calc_ref( qr_ens[ii] )

#Truncate the computed reflectivity values so there are no value below 
#the selected threshold.
dbz_ens[ dbz_ens < dbz_tr ] = dbz_tr

dbz_a_ens_temp = dbz_ens 
qr_a_ens_temp  = qr_ens

qr_ens_temp_evol = np.zeros( (ens_size , len(alpha)+1 ) )
qr_ens_evol = np.zeros( (ens_size , 2 ) )
qr_ens_temp_evol[:,0] = qr_ens
qr_ens_evol[:,0] = qr_ens


for it in range( alpha.size ) :
    dbz_a_ens_temp , qr_a_ens_temp = cf.calc_etkf_filter_update( dbz_a_ens_temp , qr_a_ens_temp , dbz_obs , R * alpha[it] )
    qr_ens_temp_evol[:,it+1] = np.copy( qr_a_ens_temp )

dbz_a_ens , qr_a_ens = cf.calc_etkf_filter_update( dbz_ens , qr_ens , dbz_obs , R )
qr_ens_evol[:,1] = qr_a_ens

pf_w = np.exp(-( dbz_ens - dbz_obs )**2/R  )
pf_w = pf_w / np.sum( pf_w)
 

dbz_a_ens[ dbz_a_ens < dbz_tr ] = dbz_tr
    


fig , axs = plt.subplots( 1 , 3 , figsize=(18,5) , sharey = False )

axs[0].plot( 1.0e4*qr_ens , dbz_ens , 'bo' ,markersize=5.0,alpha=0.05)
axs[0].plot( 1.0e4*qr_a_ens , dbz_a_ens , 'ro' ,markersize=5.0,alpha=0.05)
axs[0].plot(1.0e4*qr_true,dbz_true,'ok',markersize=10.0)
axs[0].plot(1.0e4*qr_true,dbz_obs,'og',markersize=10.0)
axs[0].set_xlabel('Graupel specific concentration')
axs[0].set_ylabel('Reflectivity')

axs[0].plot(np.mean(1.0e4*qr_ens),np.mean(dbz_ens),'ob',markersize=10.0)
axs[0].plot(np.mean(1.0e4*qr_a_ens),np.mean(dbz_a_ens),'or',markersize=10.0)
axs[0].plot(np.mean(1.0e4*qr_a_ens_temp),np.mean(dbz_a_ens_temp),'om',markersize=10.0)
axs[0].set_title('(a)')

qrf_hist , qrf_bins = np.histogram(qr_ens,bins=np.arange(0.0,4.0e-4,0.2e-4))
qrf_bins_plot= 0.5 * ( qrf_bins[0:-1] + qrf_bins[1:] )
axs[1].plot( 1.0e4*qrf_bins_plot , qrf_hist/np.sum(qrf_hist) , 'b-' ,label='Prior')

qra_hist , qra_bins = np.histogram(qr_a_ens,bins=np.arange(0.0,4.0e-4,0.2e-4))
qra_bins_plot= 0.5 * ( qra_bins[0:-1] + qra_bins[1:] )
axs[1].plot( 1.0e4*qra_bins_plot , qra_hist/np.sum(qra_hist) , 'r-' ,label='Posterior EnKF')

qrat_hist , qrat_bins = np.histogram(qr_a_ens_temp,bins=np.arange(0.0,4.0e-4,0.2e-4))
qrat_bins_plot= 0.5 * ( qrat_bins[0:-1] + qrat_bins[1:] )
axs[1].plot( 1.0e4*qrat_bins_plot , qrat_hist/np.sum(qrat_hist) , 'm-' ,label='Posterior EnKF-T3')

qrapf_hist , qrapf_bins = np.histogram(qr_ens,weights=pf_w,bins=np.arange(0.0,4.0e-4,0.2e-4))
qrapf_bins_plot= 0.5 * ( qrapf_bins[0:-1] + qrapf_bins[1:] )
axs[1].plot( 1.0e4*qrapf_bins_plot , qrapf_hist/np.sum(qrapf_hist) , 'k--' ,label='Posterior PF')
axs[1].set_xlabel('Graupel specific concentration')
axs[1].set_ylabel('Density')
axs[1].legend(fontsize=12)
axs[1].set_title('(b)')

#Select some particles. 
sel_particles = ( np.random.rand( 20 ) * ens_size ).astype(int)
pseudo_times = np.zeros( len(alpha) + 1 )
pseudo_times[1:] = np.cumsum( (1.0/alpha)/np.sum(1.0/alpha) )
axs[2].plot( 1.0e4*qr_ens_temp_evol[sel_particles,:].T , pseudo_times ,'m--',alpha=0.5)
axs[2].plot( 1.0e4*qr_ens_evol[sel_particles,:].T , np.array([0,1]) ,'r--',alpha=0.5)
axs[2].set_xlabel('Graupel specific concentration')
axs[2].set_ylabel('Pseudo time')
axs[2].set_title('(c)')


# dbzf_hist , dbzf_bins = np.histogram(dbz_ens,bins=np.arange(-30,40,2.5))
# dbzf_bins_plot= 0.5 * ( dbzf_bins[0:-1] + dbzf_bins[1:] )
# ax_y.plot( dbzf_hist/np.sum(dbzf_hist) , dbzf_bins_plot , 'b-' )

# dbza_hist , dbza_bins = np.histogram(dbz_a_ens,bins=np.arange(-30,40,2.5))
# dbza_bins_plot= 0.5 * ( dbza_bins[0:-1] + dbza_bins[1:] )
# ax_y.plot( dbza_hist/np.sum(dbza_hist) , dbza_bins_plot , 'r-' )
