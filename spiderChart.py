import matplotlib.pyplot as plt
import numpy as np


# categories is a list of the variables which are used to label each value in the spider chart
# values is a list of lists, the nested lists are the numerical values used to plot the spider chart
# labels is the legend of the graph which shows what each plot instance is supposed to represent

#BE AWARE, for each nested value you HAVE to append the first element to the end of the list or else the code wont work
def plot(categories,values,labels):
    
    label_placement = np.linspace(start=0, stop=2*np.pi,num=len(ind), endpoint=False).tolist()
    label_placement.append(label_placement[0])
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    
    for i in range(0,len(values)):
        ax.plot(label_placement,values[i],label = labels[i])

    ax.set_xticks(label_placement[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.show()

