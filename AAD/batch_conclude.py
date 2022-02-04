import numpy as np
import sys
import os
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib
import scipy.interpolate
from AAD.Objectives.ObjFunc import IndTimeModel
from scipy.interpolate import interp1d

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

if len(sys.argv)!=2:
    sys.exit("Script needs a directory name containing the batch folders to process")
else:
    bdir = sys.argv[1]
    if not os.path.isdir(bdir):
        sys.exit('Non existent directory name'+bdir)

regrets = None
dataX = None
dataOpt = None
for (root,dirs,files) in os.walk(bdir):
    # embed()
    if "com_regrets.npy" in files:
        reg = np.load(os.path.join(root,"com_regrets.npy"))
        if regrets is None:
            regrets = reg
        else:
            regrets = np.vstack((regrets,reg))

    if "X.npy" in files:
        reg = np.load(os.path.join(root,"X.npy"))
        if(reg.shape[1]==4):
            reg[:,2] = reg[:,2]*209/525 # fix the data normalization discrepancy in the Graphene data set
            reg[:,3] = reg[:,3]*21/40 # these values were normalized wrt to Si in the data file, while the paper published normalization wrt Graphene
        if dataX is None:
            dataX = np.atleast_3d(reg)
        else:
            dataX = np.dstack((np.atleast_3d(dataX),np.atleast_3d(reg)))
    if "acq_params.npy" in files:
        reg = np.load(os.path.join(root,"acq_params.npy"))
        if(reg.shape[1]==5):
            reg[:,2] = reg[:,2]*209/525 # fix the data normalization discrepancy in the Graphene data set
            reg[:,3] = reg[:,3]*21/40 # these values were normalized wrt to Si in the data file, while the paper published normalization wrt Graphene
        if dataOpt is None:
            dataOpt = np.atleast_3d(reg)
        else:
            dataOpt = np.dstack((np.atleast_3d(dataOpt),np.atleast_3d(reg)))

model = None
M = None

if dataX.shape[1]==1:
    model = IndTimeModel(problemID="QU_DI1d") # temporarily changed to QU_GR1d
elif dataX.shape[1]==3:
    model = IndTimeModel(problemID="QU_DI")
elif dataX.shape[1]==4:
    model = IndTimeModel(problemID="QU_GR")
else:
    sys.exit('Unknown dataX type') 

r = []
# embed()
num_bins = 15
for i in range(dataX.shape[1]):
    hist = np.histogram(dataX[:,i,:],bins=num_bins,range=model.bounds[i],density=True)
    hist = (hist[0] * (model.bounds[i].max() - model.bounds[i].min())/num_bins,hist[1])
    r.append(hist)

def plot_opt_params(dataOpt,model,filename=None):
    fig, ax = plt.subplots(1,dataOpt.shape[1]-1,figsize = set_size(345))
    w = 0.05; h = 0.8;l = (1./(dataOpt.shape[1]))-w/2.; b=0.1; 
    for i in range(1,dataOpt.shape[1]):
        x = np.empty((0,))
        y = np.empty((0,))
        for j in range(dataOpt.shape[2]):
            hist = np.histogram(dataOpt[:,i,j],bins=num_bins,range=model.bounds[i-1],density=False)
            # hist = (hist[0] * (model.bounds[i].max() - model.bounds[i].min())/num_bins,hist[1])
            # hist[0].shape[0]
            # y = (hist[1][1:] + hist[1][0:hist[1].shape[0]-1])/2
            # y1 = np.linspace(model.bounds[i-1].min(),model.bounds[i-1].max(),num_bins)
            # m = scipy.interpolate.interp1d(y, hist[0],fill_value="extrapolate") #function='inverse'
            # fig, ax = plt.subplots() 
            # plt.hist(dataOpt[:,i,j],num_bins,density=True)
            # plt.plot(y1,m(y1))
            N = dataOpt[:,i,j] #- model.bounds[i-1].min())/(model.bounds[i-1].max()-model.bounds[i-1].min())
            x = np.hstack( (x,np.ones((dataOpt.shape[0],))) )
            y = np.hstack((y,N))
        hh = ax[i-1].hist2d(x,y,num_bins,density=False) # np.reshape(
        ax[i-1].set_xlim(0.99,1.01) 
        ax[i-1].set_ylabel(model._search_domain[i-1]['name'])
        ax[i-1].xaxis.set_visible(False)
        w = 0.05; h = 0.8;l = (i*1./(dataOpt.shape[1]))-w/2.; b=0.1;
        ax[i-1].set_position([l,b,w,h])
    cbar = fig.colorbar(hh[3],ticks=[0,hh[0].max()])
    cbar.ax.set_yticklabels(['0','1'])
    ax[-1].set_position([l,b,w,h])
    if filename!=None:
        # embed()
        plt.savefig(filename , dpi=300)
    # plt.tight_layout()
# def plot_param_distribs(dataX,model):
#     # ax.invert_yaxis()
#     # ax.xaxis.set_visible(False)
#     # fig, ax = plt.subplots() 
#     cmap = plt.cm.get_cmap('viridis')
#     # ax.set_xlim(0, np.sum(data, axis=1).max())
#     x = np.empty((0,))
#     y = np.empty((0,))
#     for i in range(dataX.shape[1]):
#         for j in range(dataX.shape[2]):
#             hist = np.histogram(dataX[:,i,j],bins=num_bins,range=model.bounds[i],density=True)
#             # hist = (hist[0] * (model.bounds[i].max() - model.bounds[i].min())/num_bins,hist[1])
#             # hist[0].shape[0]
#             y = (hist[1][1:] + hist[1][0:hist[1].shape[0]-1])/2
#             y1 = np.linspace(model.bounds[i].min(),model.bounds[i].max(),num_bins)
#             m = scipy.interpolate.Rbf(y, hist[0],function='gaussian') #,fill_value="extrapolate",kind='cubic'
#             fig, ax = plt.subplots() 
#             plt.hist(dataX[:,i,j],num_bins,density=True)
#             plt.plot(y1,m(y1))
#             N = np.log((dataX[:,i,j] - model.bounds[i].min())/(model.bounds[i].max()-model.bounds[i].min()))
#             x = np.hstack((x,np.ones((dataX.shape[0],))+j*0.1+i*2))
#             y = np.hstack((y,N))
#         break
#     # fig, ax = plt.subplots() 
#     # hh = ax.hist2d(x,y,num_bins,density=True) # np.reshape(
#     # fig.colorbar(hh[3], ax=ax)
#     plt.show(block=False)
#     embed()
        # break
        # plt.colorbar( matplotlib.cm.ScalarMappable(cmap=cmap),cax=ax,label=model._search_domain[i]['name'] )
        # rects = data_hists[i]

# plot_param_distribs(dataX,model)
# plt.show()
plot_opt_params(dataOpt,model,os.path.join(bdir,"params.png"))
plt.show(block=False)
print(np.mean(dataOpt[:,1:,0],axis=0))
print(model.x_opt_true)
# embed()
# dX = dataX - model.x_opt_true
# n  = np.linalg.norm(dX,ord=2,axis=1)
# embed()

# ndim = len(model._search_domain)
# num_grid = int((100**(np.sqrt(ndim))))
# xs = np.empty((num_grid,ndim))
# for i in range(ndim):
#     xs[:,[i]] = np.random.uniform(low=model._search_domain[i]['domain'][0],high=model._search_domain[i]['domain'][1]*(1+1e-3),size=(num_grid,1))
# # embed()
# ys = model.evaluate_true(xs)
# std_ys = np.std(ys)
# mean_ys = np.mean(ys)

# def plot_spread(dataX,model,filename):
#     rgb_seq_len = dataX[0,:,0].shape[0]
#     seq = np.arange(rgb_seq_len*1.0)/rgb_seq_len
#     seq = seq.reshape((rgb_seq_len,1))
#     rseq = 1.0-seq
#     ones = np.ones((rgb_seq_len,1))
#     zeros = np.zeros((rgb_seq_len,1))
#     cols = np.hstack((seq**3,zeros,zeros,seq**4))
#     cols1 = np.vstack((cols,)*dataX.shape[0])
#     if dataX.shape[2]==4:
#         plt.figure()
#         # embed()
#         plt.scatter(dataX[:,:,0].reshape((1,-1)),dataX[:,:,1].reshape(1,-1),s=2,c=cols1)
#         plt.scatter(model.x_opt_true[0],model.x_opt_true[1],s=8,c='g')
#         if filename!=None:
#             plt.savefig(filename+"01.png")
#         plt.figure()
#         plt.scatter(dataX[:,:,2].reshape((1,-1)),dataX[:,:,3].reshape(1,-1),s=2,c=cols1)
#         plt.scatter(model.x_opt_true[2],model.x_opt_true[3],s=8,c='g')
#         if filename!=None:
#             plt.savefig(filename+"23.png")
#     if dataX.shape[2]==3:
#         plt.figure()
#         plt.scatter(dataX[:,:,0].reshape((1,-1)),dataX[:,:,1].reshape(1,-1),s=2,c=cols1)
#         plt.scatter(model.x_opt_true[0],model.x_opt_true[1],s=8,c='g')
#         if filename!=None:
#             plt.savefig(filename+"01.png")
#         plt.figure()
#         plt.scatter(dataX[:,:,1].reshape((1,-1)),dataX[:,:,2].reshape(1,-1),s=2,c=cols1)
#         plt.scatter(model.x_opt_true[1],model.x_opt_true[2],s=8,c='g',marker="*")
#         if filename!=None:
#             plt.savefig(filename+"12.png")
#     if dataX.shape[2]==1:
#         plt.figure()
#         plt.scatter(dataX[:,:,0].reshape((1,-1)),np.zeros(dataX[:,:,0].reshape((1,-1)).shape),s=2,c=cols1)
#         plt.scatter(model.x_opt_true[0],0.,s=8,c='g')
#         if filename!=None:
#             plt.savefig(filename+".png")
def plot_regret(regret_array,run,filename=None,yscale='log'):
    plt.figure(figsize=set_size(345))
    regret_array = regret_array/(model.f_opt)
    # regret_array= np.log(regret_array)
    x_min = np.min(regret_array[0:run+1,:],axis=0)
    x_max = np.max(regret_array[0:run+1,:],axis=0)
    x_mean = np.median(regret_array[0:run+1,:],axis=0)
    x_var = np.var(regret_array[0:run+1,:],axis=0)
    for i in range(run):
        if(i==0):
            plt.plot(regret_array[i,:],marker='.',label="all cases",ms=3,c='tan',ls='')
        else:
            plt.plot(regret_array[i,:],marker='.',ms=3,c='tan',ls='') #alpha=0.3
    skip_step = 1
    x_range = np.arange(1,x_mean.shape[-1]+1,skip_step);
    x_range_plot = np.linspace(1,x_mean.shape[-1],x_mean.shape[-1]*5);
    # cubic_interploation_model_min = interp1d(x_range, x_min, kind = "cubic")
    # cubic_interploation_model_max = interp1d(x_range, x_max, kind = "cubic")
    # cubic_interploation_model_mean = interp1d(x_range, x_mean, kind = "cubic")
    plt.plot(x_min,marker='.',label="best case",ms=3,ls='-',c='blue') #
    plt.plot(x_mean,marker='.',label="median case",c='red',ms=3,ls='-') 
    plt.plot(x_max,marker='1',label="worst case",c='forestgreen',ms=3,ls='-') #c='tab:orange',
    # plt.plot(x_range_plot,cubic_interploation_model_min(x_range_plot),marker='',ms=3,ls='-',c='blue') #
    # plt.plot(x_range_plot,cubic_interploation_model_mean(x_range_plot),marker='',c='red',ms=3,ls='-') 
    # plt.plot(x_range_plot,cubic_interploation_model_max(x_range_plot),marker='',c='forestgreen',ms=3,ls='-') #c='tab:orange',
    plt.gca().set_xscale('log')
    plt.gca().set_yscale(yscale)
    h = plt.ylabel(r'$\frac{\Delta f_{opt}}{\sigma_{\eta}}$',rotation=0,fontsize=16)
    # h.set_rotation(0); 
    plt.gca().set_xlabel(r'$iterations$')
    plt.gca().yaxis.set_label_coords(-0.12,0.4)
    # plt.gca().set_yscale('log')
    plt.grid(True)
    # Plot legend.
    lgnd = plt.legend(loc="upper right", numpoints=1, fontsize=8)
    #change the marker size manually for both lines
    lgnd.legendHandles[0]._legmarker.set_markersize(6)
    lgnd.legendHandles[1]._legmarker.set_markersize(6)
    lgnd.legendHandles[2]._legmarker.set_markersize(6)
    lgnd.legendHandles[3]._legmarker.set_markersize(6)
    # plt.legend(loc='upper right')
    plt.tight_layout()
    if filename!=None:
        # embed()
        plt.savefig(filename,dpi=300)
    


if regrets is not None:
    np.save(os.path.join(bdir,"regrets"+str(regrets.shape[0])),regrets)
    plot_regret(regrets,regrets.shape[0],os.path.join(bdir,"regrets1.png"),'linear')
    plot_regret(regrets,regrets.shape[0],os.path.join(bdir,"regrets1_log.png"),'log')

# embed()
# if model is not None:
#     plot_regret(n,n.shape[0],os.path.join(bdir,"dx_norm.png"))
    # plot_spread(dataX,model,os.path.join(bdir,"xbest_scatter"))
    # if root[0:len(bdir)+1+5] == os.path.join(bdir,"batch"):
    #     print("Running on "+root)