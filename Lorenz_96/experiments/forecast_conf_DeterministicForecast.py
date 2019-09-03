import numpy as np

#=================================================================
# GENERAL SECTION
#=================================================================

GeneralConf=dict()

GeneralConf['ExpName']='DeterministicForecast'                           #Experiment name.
GeneralConf['DataPath']='./data/forecast/'                            #Data output path
GeneralConf['FigPath']='./figs/forecast/'                             #Figures output path
GeneralConf['RunSave']=True                                           #Save the output.
GeneralConf['RunPlot']=True                                           #Plot Diagnostics.
GeneralConf['OutFile']='Forecast_' + GeneralConf['ExpName'] + '.npz'  #Output file containing the forecasts.

#File with the initial conditions
GeneralConf['AssimilationFile']='./data/Assimilation/Assimilation' + GeneralConf['ExpName'] + '.npz'
#File with the nature run (for forecast verification)
GeneralConf['NatureFile']      ='./data/Nature/NatureTwoScales.npz'

#=================================================================
#  FORECAST SECTION :
#=================================================================

ForConf=dict()

ForConf['FreqOut'] = 4                               #Forecast output frequency (in number of time steps)

ForConf['ForecastLength'] = 4 * 50                   #Maximum forecast lead time (in number of time steps)

ForConf['AnalysisSpinUp'] = 0                        #Analysis cycles to skip befor running the first forecast.

ForConf['NEns']=1                                    #Number of ensemble mebers for the nature run. (usually 1)


#=================================================================
# MODEL SECTION : 
#=================================================================

ModelConf=dict()

#General model section

ModelConf['nx'] =  40                                   #Number of large-scale state variables
ModelConf['dt']  =0.005                                #Time step for large-scale variables (do not change)
#Forcing section
ModelConf['Coef']=np.array([19.2,-0.8])                         #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... ) 
ModelConf['NCoef']=np.size(ModelConf['Coef'])           #Get the total number of coefs.

#Space dependent parameter
ModelConf['FSpaceDependent']=False                      #If the forcing parameters will depend on the location.
ModelConf['FSpaceAmplitude']=np.array([1])              #Amplitude of space variantions (for each coefficient)
ModelConf['FSpaceFreq']     =np.array([1])              #Use integers >= 1

#Parameter random walk          
ModelConf['EnablePRF']=False                            #Activate Parameter random walk
ModelConf['CSigma']=np.array([0])                       #Parameter random walk sigma
ModelConf['CPhi'  ]=1.0                                 #Parameter random walk phi

#State random forcing parameters
ModelConf['EnableSRF']=False                            #Activate State random forcing.
ModelConf['XSigma']=0.0                                 #Amplitude of the random walk
ModelConf['XPhi'  ]=1.0                                 #Time autocorrelation parameter

ModelConf['XLoc'  ]=np.arange(1,ModelConf['nx']+1)      #Location of model grid points (1-nx)

#Two scale model parameters
ModelConf['TwoScaleParameters']=np.array([10,10,0])     #Small scale and coupling parameters C , B and Hint
                                                        #Set Hint /= 0 to enable two scale model                                              
ModelConf['nxss']= ModelConf['nx'] * 32                  #Number of small scale variables
ModelConf['dtss']= ModelConf['dt'] / 10                  #Time step increment for the small scale variables
