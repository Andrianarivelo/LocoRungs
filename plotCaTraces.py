import numpy as np
import matplotlib.pyplot as plt
import pickle

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

##########################################################################################
def layoutOfPanel(ax,xLabel=None,yLabel=None,Leg=None,xyInvisible=[False,False]):


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #
    if xyInvisible[0]:
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
    else:
        ax.spines['bottom'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
    #
    if xyInvisible[1]:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')


    if xLabel != None :
        ax.set_xlabel(xLabel)

    if yLabel != None :
        ax.set_ylabel(yLabel)

    if Leg != None :
        ax.legend(loc=Leg[0], frameon=False)
        if len(Leg)>1 :
            legend = ax.get_legend()  # plt.gca().get_legend()
            ltext = legend.get_texts()
            plt.setp(ltext, fontsize=Leg[1])

##########################################################################

caTriggeredAverages = pickle.load(open('caSwingPhaseTriggeredAverages.p', 'rb'))
timeAxis = np.linspace(-0.4,0.6,(0.6+0.4)/0.02+1)

for nDays in range(len(caTriggeredAverages)):
    print(caTriggeredAverages[nDays][0])
    caTraces = caTriggeredAverages[nDays][3]
    dims = np.shape(caTraces)
    nSquared = np.sqrt(dims[1])

    nSquaredN = int(nSquared+1)
    # figure #################################
    fig_width = 25  # width in inches
    fig_height = 25  # height in inches
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
    gs.update(wspace=0.15, hspace=0.15)

    # possibly change outer margins of the figure
    plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.05)

    # sub-panel enumerations
    plt.figtext(0.06, 0.96, '%s recording, %s ROIs' % (caTriggeredAverages[nDays][0],dims[1]) , clip_on=False, color='black', size=14)
    # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

    # create panels #######################################################
    #gssub0 = gridspec.GridSpecFromSubplotSpec(nDays, 8, subplot_spec=gs[0], hspace=0.2)
    axL = []
    col = ['C1','C2','C3','C4']
    for n in range(nSquaredN*nSquaredN):
        if n==dims[1]:
            break
        ax = plt.subplot(gs[n])
        ax.axvline(x=0,ls='--',c='0.8')
        ax.axhline(y=0,ls='--',c='0.8')
        for i in range(4):
            ax.plot(timeAxis, caTraces[i][n][0], lw=2)
            ax.plot(timeAxis,caTraces[i][n][0],lw=2)
        ax.set_ylim(-0.9,1.5)
        layoutOfPanel(ax)
        #axL[n].append(ax)
    plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])
    # define vertical and horizontal spacing between panels
    #plt.show()

