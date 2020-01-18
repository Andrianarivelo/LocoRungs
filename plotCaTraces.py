import numpy as np
import matplotlib.pyplot as plt
import pickle

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

caTriggeredAverages = pickle.load(open('caSwingPhaseTriggeredAverages.p', 'rb'))
timeAxis = np.linspace(-0.2,0.6,(0.6+0.2)/0.02+1)

for nDays in range(len(caTriggeredAverages)):

    caTraces = caTriggeredAverages[nDays][3]
    dims = np.shape(caTraces)
    nSquared = np.sqrt(dims[1])

    nSquaredN = int(nSquared+1)
    # figure #################################
    fig_width = 25  # width in inches
    fig_height = 22  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
              'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
              'xtick.major.size': 4  # major tick size in points
              # 'edgecolor' : None
              # 'xtick.major.size' : 2,
              # 'ytick.major.size' : 2,
              }
    rcParams.update(params)

    # set sans-serif font to Arial
    rcParams['font.sans-serif'] = 'Arial'

    # create figure instance
    fig = plt.figure()

    # define sub-panel grid and possibly width and height ratios
    gs = gridspec.GridSpec(nSquaredN, nSquaredN,  # ,
                           # width_ratios=[1.2,1]
                           #height_ratios=[10,4]
                           )
    # define vertical and horizontal spacing between panels
    gs.update(wspace=0.1, hspace=0.1)

    # possibly change outer margins of the figure
    plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.05)

    # sub-panel enumerations
    plt.figtext(0.06, 0.96, 'days of recordings' , clip_on=False, color='black', size=14)
    # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

    # create panels #######################################################
    #gssub0 = gridspec.GridSpecFromSubplotSpec(nDays, 8, subplot_spec=gs[0], hspace=0.2)
    axL = []
    for n in range(nSquaredN*nSquaredN):
        ax = plt.subplot(gs[n])
        if n==dims[1]:
            break
        for i in range(4):
            ax.plot(timeAxis,caTraces[i][n])

        #axL[n].append(ax)

    # define vertical and horizontal spacing between panels
    plt.show()

