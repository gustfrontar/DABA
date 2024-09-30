import numpy as np

#=================================================================

#=================================================================

GeneralConf=dict()

GeneralConf['ExpName']='ImperfectInitialConditions'                       #Experiment name.
GeneralConf['DataPath']='./data/sensitivity/'                           #Data output path
GeneralConf['FigPath']='./figs/sensitivity/'                            #Figures output path
GeneralConf['RunSave']=True                                             #Save the output
GeneralConf['OutFile']='Sensitivity' + GeneralConf['ExpName'] + '.npz'  #Output file
GeneralConf['RunPlot']=True                                             #Plot Diagnostics

#Obs data, obs configuration and nature run configuration are stored
#in this file.
GeneralConf['AssimilationFile']='./data/Assimilation/AssimilationLETKF_ImperfectModel.npz'
GeneralConf['NatureFile']='./data/Nature/NatureConstantParameter.npz'

#=================================================================
# MODEL SECTION : 
#=================================================================
#General model section

ModelConf=dict()

#General model section

ModelConf['nx'] =  40                                   #Number of large-scale state variables
ModelConf['dt']  =0.0125                                #Time step for large-scale variables (do not change)

#Model parameters section
ModelConf['Coef']=np.array([8,0,0])                     #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... ) 
ModelConf['NCoef']=ModelConf['Coef'].size

#Space dependent parameter

ModelConf['FSpaceDependent']=False                      #If the forcing parameters will depend on the location.
ModelConf['FSpaceAmplitude']=np.array([1,1,1])          #Amplitude of space variantions (for each coefficient)
ModelConf['FSpaceFreq']     =1                          #Use integers >= 1

#Parameter random walk          

ModelConf['EnablePRF']=False                            #Activate Parameter random walk
ModelConf['CSigma']=np.array([0,0,0])                   #Parameter random walk sigma
ModelConf['CPhi'  ]=1.0                                 #Parameter random walk phi

#State random forcing parameters

ModelConf['EnableSRF']=False                            #Activate State random forcing.
ModelConf['XSigma']=0.0                                 #Amplitude of the random walk
ModelConf['XPhi'  ]=1.0                                 #Time autocorrelation parameter

ModelConf['XLoc'  ]=np.arange(1,ModelConf['nx']+1)      #Location of model grid points (1-nx)


#=================================================================
#  SENSITIVITY SECTION :
#=================================================================

SensitivityConf=dict()

SensitivityConf['PIndex']=0   #Select the parameter that will be changed.

SensitivityConf['PMax'] = np.array([11.0])    #Maximum parameter value
SensitivityConf['PMin'] = np.array([5.0])    #Minimum parameter value
   
SensitivityConf['PRes'] = np.array([0.2])    #Parameter sampling resolution.

SensitivityConf['PVals'] = np.arange(SensitivityConf['PMin'],SensitivityConf['PMax']+SensitivityConf['PRes'],SensitivityConf['PRes'])  #Parameter values to be used to compute model sensitivity.

SensitivityConf['NP'] = SensitivityConf['PVals'].size   #Number of different parameter values to be explored.


SensitivityConf['AnalysisSpinUp'] = 200                 #Analysis cycles to skip befor running the first forecast.

SensitivityConf['UseNatureRunAsIC'] = False             #If true the nature run will be used as IC for the forecasts (perfect initial conditions)

SensitivityConf['ForecastLength'] = 5*4                #Length of the forecast that will be used to asses model sensitivity






        
        
        
        















