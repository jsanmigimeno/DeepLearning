from matplotlib import pyplot as plt

def plotGraph(loss, lossType, title, secondMetric='', secondMetricType=''):
    plt.rcParams["font.family"] = 'Times New Roman'
    fig, ax1 = plt.subplots(figsize=(6, 3.75), dpi=100)
    plt.title(title, fontsize=20)

    plt.grid()
    N = len(loss[0])
    lns1 = ax1.plot(loss[0], color="tab:blue", label="Training", zorder=100)
    lns2 = ax1.plot(loss[1], color="tab:blue", linestyle="--", label="Validation", zorder=75)
    lns = lns1 + lns2
    # Set x-axis format
    ax1.set_xlabel("Epoch", fontsize=18)
    ax1.set_xticks([0] + list(range(4, N+1, 5)))
    ax1.set_xticklabels([1] + list(range(5, N+1, 5)), fontsize=14)
    # ax1.set_xticks([0] + list(range(9, 85, 10)))
    # ax1.set_xticklabels([1] + list(range(10, 86, 10)), fontsize=14)

    # Set y-axis format
    ax1.set_ylabel("Loss" + ", " + lossType, color="tab:blue", fontsize=18)
    plt.yticks(fontsize=14)

    if secondMetric!='':
        ax2 = ax1.twinx()
        ax2.set_ylabel(secondMetricType, color="tab:red", fontsize=18)
        lns3 = ax2.plot(secondMetric[0], color="tab:red", label="Tr. " + secondMetricType, zorder=50)
        lns4 = ax2.plot(secondMetric[1], color="tab:red", linestyle="--", label="Val. " + secondMetricType, zorder=25)
        lns = lns + lns3 + lns4
        plt.yticks(fontsize=14)


    labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs, fontsize=11)
    #ax1.legend(lns, labs, fontsize=11, bbox_to_anchor=(0.025, 0.85, 0.5, .102), loc=3, ncol=2, mode="expand", borderaxespad=0)
    ax1.legend(lns, labs, fontsize=11, loc=1, ncol=2)

    fig.tight_layout()
    plt.show()
    # plt.plot(loss)
    # plt.plot(mse)
    # plt.plot(vloss)
    # plt.plot(vmse)