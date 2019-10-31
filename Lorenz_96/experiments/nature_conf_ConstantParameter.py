import numpy as np

#=================================================================
# GENERAL SECTION : 
#=================================================================

GeneralConf=dict()

GeneralConf['ExpName']='ConstantParameter'                #Experiment name.
GeneralConf['DataPath']='./data/Nature/'                  #Data output path
GeneralConf['FigPath']='./figs/Nature/'                   #Figures output path
GeneralConf['NatureFileName']='Nature' + GeneralConf['ExpName'] + '.npz'

#=================================================================
# MODEL SECTION : 
#=================================================================

ModelConf=dict()

#General model section

ModelConf['nx'] =  40                                   #Number of large-scale state variables
ModelConf['dt']  =0.0125                                #Time step for large-scale variables (do not change)
#Forcing section
ModelConf['Coef']=np.array([8])                         #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... ) 
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
ModelConf['nxss']= ModelConf['nx'] * 8                  #Number of small scale variables
ModelConf['dtss']= ModelConf['dt'] / 5                  #Time step increment for the small scale variables

#=================================================================
# NATURE RUN SECTION : 
#=================================================================

NatureConf= dict()

NatureConf['NEns']=1                               #Number of ensemble mebers for the nature run. (usually 1)

NatureConf['RunSave']=True                         #Save nature run
NatureConf['RunPlot']=True                         #Plot nature run

NatureConf['SPLength']=40                          #Spin up length in model time units (1 model time unit app. equivalent to 5 day time in the atmosphere)
NatureConf['Length']=40                            #Nature run length in model time units (1 model time unit app. equivalent to 5 day time in the atmosphere)

#=================================================================
# OBSERVATION CONFIGURATION SECTION : 
#=================================================================

ObsConf= dict()

ObsConf['Freq']=4                                    #Observation frequency in number of time steps (will also control nature run output frequency)

#Observation location
ObsConf['NetworkType']='regular'                     #Observation network type: REGULAR, RANDOM, FROMFILE

ObsConf['SpaceDensity']=1.0                          #Observation density in space. Usually from [0-1] but can be greather than 1.
ObsConf['TimeDensity']=1                             #Observation density in time.  Usually from [0-1] but can be greather than 1. 
                                                     #Do not use ObsTimeDensity to change observation frequency for REGULAR obs, use
                                                     #ObsFreq instead.

#Set the diagonal of R
ObsConf['Error']=0.2                                 #Constant observation error.

#Set the systematic observation error
ObsConf['Bias']=0.0                                  #Constant Systematic observation error.

#Set observation type 1-Observe X 
ObsConf['Type']=1                                    #Observation type (1 observ x, 2 observe x**2)





