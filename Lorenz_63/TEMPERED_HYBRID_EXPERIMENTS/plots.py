import numpy as np
import matplotlib.pyplot as plt

def plot_xyz_evol( x_ens_evol , xplot , yplot , zplot , xf_den , xf_xden , xf_yden , xf_zden , xa_den , xa_xden , xa_yden , xa_zden ,
                                like_den , xt , yo , title , fig_name_prefix ,  bst , numtrans , metrics ) :

   #Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
   Ntimes = x_ens_evol.shape[2]
   EnsSize = x_ens_evol.shape[1]
   pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
   pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

   plt.figure(figsize=(10,4))
   plt.subplot(1,3,1)
   plt.plot(x_ens_evol[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
   plt.plot(x_ens_evol[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
   plt.plot(x_ens_evol[0,:,:].transpose(),pseudo_time_mat, '--',color='#C0C0C0',linewidth=0.5)             #Evolution
   plt.plot(xplot,xf_xden/2.0,color='#842020',linewidth=4.0,label='Prior Den')
   plt.plot(xplot,xa_xden/2.0,color='#000066',linewidth=4.0,label='Posterior Den')
   plt.title('X')
   plt.plot(xt[0],1,'ks',label='True')
   plt.subplot(1,3,2)
   plt.plot(x_ens_evol[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
   plt.plot(x_ens_evol[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
   plt.plot(x_ens_evol[1,:,:].transpose(),pseudo_time_mat, '--',color='#C0C0C0',linewidth=0.5)             #Evolution
   plt.plot(yplot,xf_yden/2.0,color='#842020',linewidth=4.0,label='Prior Den')
   plt.plot(yplot,xa_yden/2.0,color='#000066',linewidth=4.0,label='Posterior Den')
   plt.plot(xt[1],1,'ks',label='True')
   plt.title('Y')
   plt.subplot(1,3,3)
   plt.plot(x_ens_evol[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
   plt.plot(x_ens_evol[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
   plt.plot(x_ens_evol[2,:,:].transpose(),pseudo_time_mat, '--',color='#C0C0C0',linewidth=0.5)             #Evolution
   plt.plot(zplot,xf_zden/2.0,color='#842020',linewidth=4.0,label='Prior Den')
   plt.plot(zplot,xa_zden/2.0,color='#000066',linewidth=4.0,label='Posterior Den')
   plt.plot(xt[2],1,'ks',label='True')
   plt.title('Z')
   plt.suptitle(title)
   plt.legend()
   plt.savefig(fig_name_prefix + str(bst) + '_' + str(numtrans) + '_1dparticle_evolution.png')
   #plt.close()
   plt.show()


   plt.figure()
   plt.plot( np.transpose(x_ens_evol[0,:,:]) , np.transpose( x_ens_evol[1,:,:] ) , '--',color='#C0C0C0',linewidth=0.5,zorder=1)

   plt.plot(x_ens_evol[0,:,0],x_ens_evol[1,:,0],'ro',label='Prior Ens',zorder=2,alpha=0.5)  #Prior
   plt.plot(x_ens_evol[0,:,-1],x_ens_evol[1,:,-1],'bo',label='Posterior Ens',zorder=3,alpha=0.5)  #Posterior
   plt.plot(xt[0],xt[1],'ks',label='True')

   denmax=np.max(xf_den)
   dden=denmax/3.0
   levels=np.arange(0.001,denmax,dden)
   plt.contour(xplot,yplot,np.transpose(xf_den),colors=('#842020'),linewidths=3.0,levels=levels,zorder=5)
   denmax=np.max(xa_den)
   dden=denmax/3.0
   levels=np.arange(0.001,denmax,dden)
   plt.contour(xplot,yplot,np.transpose(xa_den),colors=('#000066'),linewidths=3.0,levels=levels,zorder=6)
   denmax=np.max(like_den)
   dden=denmax/3.0
   levels=np.arange(0.001,denmax,dden)
   plt.contour(xplot,yplot,np.transpose(like_den),colors='k',linewidths=2.0,levels=levels,zorder=4)

   text='kld='+str(np.round(100.0*metrics['kld_a'])/100.0)+' ('+str(np.round(100.0*metrics['kld_f'])/100.0)+')'
   plt.annotate(text,(0.65,0.6),xycoords='figure fraction')
   text='cov dist='+str(np.round(100.0*metrics['dist_covar_a'])/100.0)+' ('+str(np.round(100.0*metrics['dist_covar_f'])/100.0)+')'
   plt.annotate(text,(0.65,0.55),xycoords='figure fraction')
   text='var dist='+str(np.round(100.0*metrics['dist_var_a'])/100.0)+' ('+str(np.round(100.0*metrics['dist_var_f'])/100.0)+')'
   plt.annotate(text,(0.65,0.5),xycoords='figure fraction')
   text='var bias='+str(np.round(100.0*metrics['bias_var_a'])/100.0)+' ('+str(np.round(100.0*metrics['bias_var_f'])/100.0)+')'
   plt.annotate(text,(0.65,0.45),xycoords='figure fraction')
   text='mean bias='+str(np.round(100.0*metrics['bias_a'])/100.0)+' ('+str(np.round(100.0*metrics['bias_f'])/100.0)+')'
   plt.annotate(text,(0.65,0.40),xycoords='figure fraction')
   text='mean rmse='+str(np.round(100.0*metrics['rmse_a'])/100.0)+' ('+str(np.round(100.0*metrics['rmse_f'])/100.0)+')'
   plt.annotate(text,(0.65,0.35),xycoords='figure fraction')






   plt.title(title)
   plt.legend()

   plt.savefig(fig_name_prefix + str(bst) + '_' + str(numtrans) + '_2d_particle_evolution_xy.png')
   #plt.close()
   plt.show()

