import pickle
import numpy
import gzip
from matplotlib import pyplot as plt

# import keras

## Denoise
# # Open denoise files individually
# ids = range(-6, 40)

# historyObjs = []

# for id in ids:
#     with open('./Denoise Model/Fast Model/denoise_history_' + str(id) +'.pkl', 'rb') as f:
#         hist = pickle.load(f)
#         historyObjs.append(hist)

# # Open history objects list
# with open('./Denoise Model/Fast Model/groupedHistObjs.pkl', 'wb') as f:
#     pickle.dump(historyObjs, f)

# # Open history objects list
# with open('./Denoise Model/Fast Model/groupedHistObjs.pkl', 'rb') as f:
#     historyObjs = pickle.load(f)

# loss = []
# mae = []
# mse = []
# vloss = []
# vmae = []
# vmse = []

# for obj in historyObjs:
#     loss.append(obj.history['loss'])
#     mae.append(obj.history['mean_absolute_error'])
#     mse.append(obj.history['mean_squared_error'])
#     vloss.append(obj.history['val_loss'])
#     vmae.append(obj.history['val_mean_absolute_error'])
#     vmse.append(obj.history['val_mean_squared_error'])

# with open('./Denoise Model/Fast Model/extractedLosses.pkl', 'wb') as f:
#     pickle.dump([loss, mae, mse, vloss, vmae, vmse], f)



# Open denoise files individually
ids = range(0, 40)

loss = []
mae = []
mse = []
vloss = []
vmae = []
vmse = []

#historyObjs = []

# for id in ids:
#     with open('./Descriptor Model/descriptor_history_' + str(id) +'.pkl', 'rb') as f:
#         obj = pickle.load(f)
#         loss.append(obj.history['loss'])
#         mae.append(obj.history['mean_absolute_error'])
#         mse.append(obj.history['mean_squared_error'])
#         vloss.append(obj.history['val_loss'])
#         vmae.append(obj.history['val_mean_absolute_error'])
#         vmse.append(obj.history['val_mean_squared_error'])
#         #historyObjs.append(hist)

# # Open history objects list
# with open('./Descriptor Model/groupedHistObjs.pkl', 'wb') as f:
#     pickle.dump(historyObjs, f)

# # Open history objects list
# with open('./Descriptor Model/groupedHistObjs.pkl', 'rb') as f:
#     historyObjs = pickle.load(f)


# for obj in historyObjs:
#     loss.append(obj.history['loss'])
#     mae.append(obj.history['mean_absolute_error'])
#     mse.append(obj.history['mean_squared_error'])
#     vloss.append(obj.history['val_loss'])
#     vmae.append(obj.history['val_mean_absolute_error'])
#     vmse.append(obj.history['val_mean_squared_error'])

# with open('./Descriptor Model/extractedLosses.pkl', 'wb') as f:
#     pickle.dump([loss, mae, mse, vloss, vmae, vmse], f)




with open('../Denoise Model/Fast Model/extractedLosses.pkl', 'rb') as f:
    [loss, mae, mse, vloss, vmae, vmse] = pickle.load(f)

with open('../Descriptor Model/extractedLosses.pkl', 'rb') as f:
    [loss, mae, mse, vloss, vmae, vmse] = pickle.load(f)

# loss[1][0] = loss[1][0]+0.2
# mse[1][0] = mse[1][0]+3
# for id in range(-6,-4,1):
#     with open('../Denoise Model/Fast Model/denoise_history_' + str(id) +'.pkl', 'rb') as f:
#         hist = pickle.load(f)
#         loss.pop(id+6)
#         loss.insert(id+6, [hist[0]])
#         mse.pop(id+6)
#         mse.insert(id+6, [hist[2]])
#         #historyObjs.append(hist)

for id in range(40,80,1):
    with open('../Descriptor Model/descriptor_history2_' + str(id) +'.pkl', 'rb') as f:
        hist = pickle.load(f)
        loss.append(hist[0])
        mse.append(hist[1])
        vloss.append(hist[2])
        vmse.append(hist[3])
        #historyObjs.append(hist)

def plotGraph(loss, mae, mse, vloss, vmae, vmse, title):
    plt.rcParams["font.family"] = 'Times New Roman'
    fig, ax1 = plt.subplots(figsize=(6, 3.75), dpi=100)
    plt.title(title, fontsize=20)

    plt.grid()

    lns1 = ax1.plot(loss, color="tab:blue", label="Training", zorder=100)
    lns2 = ax1.plot(vloss, color="tab:blue", linestyle="--", label="Validation", zorder=75)

    # Set x-axis format
    ax1.set_xlabel("Epoch", fontsize=18)
    # ax1.set_xticks([0] + list(range(4, 85, 5)))
    # ax1.set_xticklabels([1] + list(range(5, 86, 5)), fontsize=14)
    ax1.set_xticks([0] + list(range(9, 85, 10)))
    ax1.set_xticklabels([1] + list(range(10, 86, 10)), fontsize=14)

    # Set y-axis format
    ax1.set_ylabel("Triplet Loss", color="tab:blue", fontsize=18)
    plt.yticks(fontsize=14)

    # ax2 = ax1.twinx()
    # ax2.set_ylabel("MSE", color="tab:red", fontsize=18)
    # lns3 = ax2.plot(mse, color="tab:red", label="Training MSE", zorder=50)
    # lns4 = ax2.plot(vmse, color="tab:red", linestyle="--", label="Validation MSE", zorder=25)
    plt.yticks(fontsize=14)



    # added these three lines
    #lns = lns1+lns2+lns3+lns4
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs, fontsize=11)
    ax1.legend(lns, labs, fontsize=11, bbox_to_anchor=(0.025, 0.85, 0.5, .102), loc=3, ncol=2, mode="expand", borderaxespad=0)

    fig.tight_layout()
    plt.show()
    # plt.plot(loss)
    # plt.plot(mse)
    # plt.plot(vloss)
    # plt.plot(vmse)


#plotGraph(loss, mae, mse, vloss, vmae, vmse, "Denoise Net Loss")
plotGraph(loss, mae, mse, vloss, vmae, vmse, "Descriptor Net Loss")
