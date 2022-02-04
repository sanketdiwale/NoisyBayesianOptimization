import numpy as np
import sys
import os
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib
import scipy.interpolate
from AAD.Objectives.ObjFunc import IndTimeModel
from scipy.interpolate import interp1d
import math
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

bdirs = []
# embed()
if len(sys.argv)<=2:
    sys.exit("Script needs atleast 2 directory names containing the batch folders to process")
else:
    for j in range(1,len(sys.argv)):
        bdirs.append( sys.argv[j] )
        if not os.path.isdir(bdirs[-1]):
            sys.exit('Non existent directory name '+bdirs[-1])

Regrets = []
# embed()
for bdir in bdirs:
    regrets = None
    for (root,dirs,files) in os.walk(bdir):
        if "com_regrets.npy" in files:
            reg = np.load(os.path.join(root,"com_regrets.npy"))
            if(reg.shape[0]>2000):
                reg = reg[-2000:,:].T
                # embed()
            if regrets is None:
                regrets = reg
            else:
                # embed()
                regrets = np.vstack((regrets,reg))
    Regrets += [regrets]

# embed()
dataX=None
for (root,dirs,files) in os.walk(bdirs[0]):
    if "X.npy" in files:
        reg = np.load(os.path.join(root,"X.npy"))
        if dataX is None:
            dataX = np.atleast_3d(reg)
        else:
            dataX = np.dstack((np.atleast_3d(dataX),np.atleast_3d(reg)))

if dataX is None:
    dataX = np.empty((0,4))

model = None


if dataX.shape[1]==1:
    model = IndTimeModel(problemID="QU_DI1d") # temporarily changed to QU_GR1d
elif dataX.shape[1]==3:
    model = IndTimeModel(problemID="QU_DI")
elif dataX.shape[1]==4:
    model = IndTimeModel(problemID="QU_GR")
else:
    sys.exit('Unknown dataX type') 
q_arr = [1,10,20,40]
q_label_arr = ['q = 1','q = 10','q = 20', 'q = 40']
q_ls = ['-','-','-','-']#[':','-.','-.','-.']
p_arr = [0,0,1,2]
p_label_arr = ['(Ueno et al.)','p = 0','p = 1\n(Huang et al.)', 'p = 2']
p_ls = ['-','-','-','-']#[':','-.','-.','-.']
p_c_arr = ['tan','red','forestgreen','blue']
c_arr = ['tan','red','forestgreen','blue']
def plot_compare(regret_arrays,run,filename=None,yscale='log',compare_type='p'):
    color_arr = p_c_arr
    label_arr = p_arr
    xscale = 'log'
    if compare_type=='p':
        color_arr = p_c_arr
        label_arr = p_label_arr
        a_ls = p_ls
    else:
        color_arr = c_arr
        label_arr = q_label_arr
        a_ls = q_ls
    plt.figure(figsize=set_size(345))
    for i, regret_array in enumerate(regret_arrays):
        if((compare_type=="p") and i==0):
            regret_array = regret_array[:,0::10] # Ueno et al uses q=1, but here we are comparing to q=10 as the default case for the figure, thus we plot the progress of the Ueno et al over batches of 10 samples to keep the plots comparable (even though Ueno et al makes a decision every single sample, i.e. with q=1)
        regret_array = regret_array/(model.f_opt)
        # regret_array= np.log(regret_array)
        x_min = np.min(regret_array[0:run+1,:],axis=0)
        x_max = np.max(regret_array[0:run+1,:],axis=0)
        x_mean = np.median(regret_array[0:run+1,:],axis=0)
        x_var = np.var(regret_array[0:run+1,:],axis=0)
        # skip_step = 1
        # x_range = np.arange(1,x_mean.shape[-1]+1,skip_step);
        # x_range = np.hstack((x_range,np.array(x_mean.shape[-1])))
        # y_range = x_mean[::skip_step] #np.hstack((x_max[::skip_step],np.array(x_max[-1])))
        # x_range_plot = np.linspace(1,x_mean.shape[-1],x_mean.shape[-1]*5);
        # embed()
        # cubic_interploation_model = interp1d(x_range, y_range, kind = "cubic")
        # plt.plot(x_min,marker='.',label="best case",ms=3,ls='') #
        # plt.plot(x_range_plot,cubic_interploation_model(x_range_plot),marker='',label="q = "+str(q_arr[i]),ms=3,ls='-',c=c_arr[i]) # c='red'
        # plt.plot(x_mean,marker='.',ms=3,ls='-',c=color_arr[i],label=compare_type+" = "+str(label_arr[i])) # c='red'
        plt.plot(x_mean,marker='.',ms=3,ls='-',c=color_arr[i],label=label_arr[i]) # c='red'
    plt.gca().set_xscale(xscale)
    plt.gca().set_yscale(yscale)
    h = plt.ylabel(r'$\frac{\Delta f_{opt}}{\sigma_{\eta}}$',rotation=0,fontsize=16)
    # h.set_rotation(0); 
    plt.gca().set_xlabel(r'$iterations$')
    plt.gca().yaxis.set_label_coords(-0.12,0.45)
    # plt.gca().set_yscale('log')
    plt.grid(True)
    # Plot legend.
    lgnd = plt.legend(loc="upper right", numpoints=1, fontsize=8)
    #change the marker size manually for both lines
    for i in range(len(regret_arrays)):
        lgnd.legendHandles[i]._legmarker.set_markersize(6)
    plt.tight_layout()
    if filename!=None:
        root,ext = os.path.splitext(filename)
        plt.savefig(root+"_med"+ext, dpi=300)
    
    plt.figure(figsize=set_size(345))
    for i, regret_array in enumerate(regret_arrays):
        if((compare_type=="p") and i==0):
            a = np.split(regret_array,200,axis=1)
            l = [np.max(x,axis=1).reshape(-1,1) for x in a]
            regret_array = np.concatenate(l,axis=1)
            # embed()
            # regret_array = regret_array[:,0::10] # Ueno et al uses q=1, but here we are comparing to q=10 as the default case for the figure, thus we plot the progress of the Ueno et al over batches of 10 samples to keep the plots comparable (even though Ueno et al makes a decision every single sample, i.e. with q=1)
        regret_array = regret_array/(model.f_opt)
        x_max = np.max(regret_array[0:run+1,:],axis=0)
        plt.plot(x_max,marker='.',ms=3,ls=a_ls[i],c=color_arr[i],label=label_arr[i]) # c='red'
    plt.gca().set_xscale(xscale)
    plt.gca().set_yscale(yscale)
    h = plt.ylabel(r'$\frac{\Delta f_{opt}}{\sigma_{\eta}}$',rotation=0,fontsize=16)
    # h.set_rotation(0); 
    plt.gca().set_xlabel(r'$iterations$')
    plt.gca().yaxis.set_label_coords(-0.12,0.45)
    # plt.gca().set_yscale('log')
    plt.grid(True)
    # Plot legend.
    lgnd = plt.legend(loc="upper right", numpoints=1, fontsize=6)
    #change the marker size manually for both lines
    for i in range(len(regret_arrays)):
        lgnd.legendHandles[i]._legmarker.set_markersize(6)
    plt.tight_layout()
    if filename!=None:
        root,ext = os.path.splitext(filename)
        plt.savefig(root+"_max"+ext, dpi=300)
    
    plt.figure(figsize=set_size(345))
    for i, regret_array in enumerate(regret_arrays):
        if((compare_type=="p") and i==0):
            regret_array = regret_array[:,0::10] # Ueno et al uses q=1, but here we are comparing to q=10 as the default case for the figure, thus we plot the progress of the Ueno et al over batches of 10 samples to keep the plots comparable (even though Ueno et al makes a decision every single sample, i.e. with q=1)
        regret_array = regret_array/(model.f_opt)
        x_var = np.var(regret_array[0:run+1,:],axis=0)
        plt.plot(x_var,marker='.',ms=3,ls='-',c=color_arr[i],label=label_arr[i]) # c='red'
    plt.gca().set_xscale(xscale)
    plt.gca().set_yscale(yscale)
    h = plt.ylabel(r'$\frac{\Delta f_{opt}}{\sigma_{\eta}}$',rotation=0,fontsize=16)
    # h.set_rotation(0); 
    plt.gca().set_xlabel(r'$iterations$')
    plt.gca().yaxis.set_label_coords(-0.12,0.45)
    # plt.gca().set_yscale('log')
    plt.grid(True)
    # Plot legend.
    lgnd = plt.legend(loc="upper right", numpoints=1, fontsize=6)
    #change the marker size manually for both lines
    for i in range(len(regret_arrays)):
        lgnd.legendHandles[i]._legmarker.set_markersize(6)
    plt.tight_layout()
    if filename!=None:
        root,ext = os.path.splitext(filename)
        plt.savefig(root+"_var"+ext, dpi=300)

    plt.figure(figsize=set_size(345))
    bar_width = 0.5; y_offset = 0; xpos =[]; lab=[]; 
    for i, regret_array in enumerate(regret_arrays):
        if((compare_type=="p") and i==0):
            regret_array = regret_array[:,0::10] # Ueno et al uses q=1, but here we are comparing to q=10 as the default case for the figure, thus we plot the progress of the Ueno et al over batches of 10 samples to keep the plots comparable (even though Ueno et al makes a decision every single sample, i.e. with q=1)
        regret_array = regret_array/(model.f_opt)
        # x_max = np.max(regret_array[0:run+1,:],axis=0)
        x_var = np.var(regret_array[0:run+1,:],axis=0)
        sample_frac = 0.35
        l = math.ceil(sample_frac*2000/10)
        if compare_type=='p':
            l = math.ceil(sample_frac*2000/10)
        else:
            l = math.ceil(sample_frac*2000/(label_arr[i]))
        m = np.max(x_var[x_var.shape[-1] - l:-1])
        plt.bar(1+i, m, bar_width, bottom=y_offset, color=color_arr[i],label=label_arr[i])
        xpos.append(1+i)
        lab.append(label_arr[i])
        # plt.plot(x_var,marker='.',ms=3,ls='-',c=color_arr[i],label=compare_type+" = "+str(label_arr[i])) # c='red'
    plt.gca().set_xscale('linear')
    plt.gca().set_yscale(yscale)
    plt.xticks(xpos, lab,
       rotation=0)
    # plt.yticks(np.arange(0, max(x)+1, 1.0)) 
    h = plt.ylabel(r'Q$\left(\frac{\Delta f_{opt}}{\sigma_{\eta}}\right)$',rotation=0,fontsize=16)
    # h.set_rotation(0); 
    plt.gca().set_xlabel(r'')
    if compare_type=='p':
        yoffset = 0.75
        xoffset = -0.2
    else:
        yoffset = 0.85
        xoffset = -0.27
    plt.gca().yaxis.set_label_coords(xoffset,yoffset)
    # plt.gca().set_yscale('log')
    plt.grid(True)
    # Plot legend.
    # lgnd = plt.legend(loc="upper right", numpoints=1, fontsize=10)
    #change the marker size manually for both lines
    # for i in range(len(regret_arrays)):
    #     lgnd.legendHandles[i]._legmarker.set_markersize(6)
    plt.tight_layout()
    if filename!=None:
        root,ext = os.path.splitext(filename)
        plt.savefig(root+"_term_var"+ext, dpi=300)

# if regrets is not None:
# np.save(os.path.join(bdirs[0],"Regrets"+str(Regrets.shape[0])),regrets)

# embed()
ptype = 'p'
if len(bdirs)==len(p_arr):
    ptype='p'
else:
    ptype='q'
plot_compare(Regrets,Regrets[0].shape[0],os.path.join(bdirs[0],"compare_"+ptype+"_log.png"),'log',ptype)
plot_compare(Regrets,Regrets[0].shape[0],os.path.join(bdirs[0],"compare_"+ptype+".png"),'linear',ptype)
plt.show(block=False)
# embed()