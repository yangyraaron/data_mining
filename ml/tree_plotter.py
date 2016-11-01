"""
customerized plotter for tree
"""
import matplotlib.pyplot as plt


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    plot a single node
    """
    create_plot.ax1.annotate(node_text, xy=parent_pt,
    xycoords='axes fraction',
    xytext=center_pt, textcoords='axes fraction',
    va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    create_plot.ax1.text(xMid, yMid, txtString)


def plot_tree(myTree, parentPt, nodeTxt):
    numLeafs = get_leaf_size(myTree)
    get_tree_depth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs))/2.0/plot_tree.totalW,\
    plot_tree.yOff)
    plot_mid_text(cntrPt, parentPt, nodeTxt)
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    secondDict = myTree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plot_tree(secondDict[key],cntrPt,str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff),
                cntrPt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD


def create_plot(tree):
    """
    create a plot
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_leaf_size(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    plot_tree.xOff = -0.5/plot_tree.totalW; plot_tree.yOff = 1.0;
    plot_tree(tree, (0.5,1.0), '')
    plt.show()


def get_leaf_size(tree):
    """
    get number of leaf nodes
    """
    size = 0
    root_cls = tree.keys()[0]
    root_node = tree[root_cls]

    for cls, node in root_node.iteritems():
        if isinstance(node, dict):
            size += get_leaf_size(node)
        else:
            size +=1

    return size


def get_tree_depth(tree):
    """
    get depth of a tree
    """
    max_depth = 0
    root_cls = tree.keys()[0]
    root_node = tree[root_cls]
    for cls, node in root_node.iteritems():
        if isinstance(node, dict):
            cur_depth = 1 + get_tree_depth(node)
        else:
            cur_depth = 1
        if cur_depth > max_depth:
            max_depth = cur_depth

    return max_depth


tree_data = [{'no surfacing': {0: 'no', 1: {'flippers': \
 {0: 'no', 1: 'yes'}}}},
 {'no surfacing': {0: 'no', 1: {'flippers': \
 {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
 ]
