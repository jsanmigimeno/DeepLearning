#%%
import plotFuncs as pF
import pickle

#%%
with open('./DenoiserDeepT/denoise_history.pkl', 'rb') as f:
    allLosses = pickle.load(f)

pF.plotGraph(allLosses[0:3], 'MSE', '', allLosses[2:4], 'MAE')