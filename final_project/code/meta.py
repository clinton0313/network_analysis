#%%
import numpy as np
from investigation import Investigation
import pickle, os, matplotlib, networkx
from matplotlib import pyplot as plt
import seaborn as sns
from time import sleep
import networkx as nx
from node2vec import Node2Vec
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean

#%% Dirs, colors, graphs
basedir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
import matplotlib.colors as mcolors

colors = [color for _, color in list(mcolors.CSS4_COLORS.items())[10:25]]

with open(os.path.join("data", "processed_data", "giant_component_crime_networks.pkl"), "rb") as infile:
    network_graphs = pickle.load(infile)
# %% Auxiliary functions
def gap_stat(data, nrefs=20, ks=range(1,11)):
    data_shape = data.shape

    top_col = data.max(axis=0)
    bot_col = data.min(axis=0)
    dists = np.matrix(np.diag(top_col-bot_col))

    rands = np.random.sample(size=(data_shape[0],data_shape[1],20))
    for i in range(nrefs):
        rands[:,:,i] = rands[:,:,i]*dists+bot_col
    
    gaps = np.zeros((len(ks),))
    for (i,k) in enumerate(ks):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        kmc, kml = kmeans.cluster_centers_, kmeans.labels_

        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(data_shape[0])])

        refdisps = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            # (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(rands[:,:,j])
            kmc, kml = kmeans.cluster_centers_, kmeans.labels_

            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(data_shape[0])])
        gaps[i] = np.mean(np.log(refdisps))-np.log(disp)
    return gaps

# %%

results = []
for set in range(3): 
    fig, ax = plt.subplots(
        6,3,
        figsize=(32,32),        
        sharex=False, 
        sharey=False, 
        gridspec_kw={'hspace': 0.3, 'wspace': 0.1, "width_ratios": [3,3,1]}
    )
    for i, crime_network in enumerate(network_graphs[(set*6):(set*6)+6]):

        # %% General plot for graph
        layout = nx.layout.spring_layout(crime_network, k = 0.5 / np.sqrt(len(crime_network.nodes)))
    
        ec = nx.eigenvector_centrality_numpy(crime_network)

        nx.draw_networkx(
            crime_network, 
            pos=layout, 
            ax=ax[i,0], 
            with_labels=False,
            alpha=0.7,
            node_color=list(ec.values()), 
            cmap="Reds", 
            edgecolors="black"
        )

        ax[i,0].set_title(
            crime_network.graph["name"] + ": Eigenvector Centrality", 
            fontsize=30
        )

        # Compute lower-embedding space
        # Precompute probabilities and generate walks
        node2vec = Node2Vec(crime_network, dimensions=16, walk_length=30, num_walks=200, workers=1)  
        # Embed nodes
        model = node2vec.fit(window=10, min_count=1, batch_words=4) 
        embeddings = model.wv.get_normed_vectors()
        lst = list(model.wv.key_to_index)
        embeddings = model.wv[lst]

        # Gap stat
        gaps = gap_stat(embeddings)
        kc = np.argmax(gaps)

        # Get kmeans
        kmeans = KMeans(n_clusters = kc, random_state=42).fit(embeddings)
        clusters = kmeans.labels_

        colrs = [colors[j] for j in clusters]

        # Plot clustered graph
        color_att = dict(zip(list(map(int, lst)), colrs))
        nx.set_node_attributes(crime_network, color_att, "color")

        nx.draw_networkx(crime_network, pos=layout, 
            ax=ax[i,1], 
            with_labels=False,
            alpha=0.7,
            # edge_color = "#d18b4f",
            linewidths=2,
            node_color = dict(crime_network.nodes.data("color")).values(),
            edgecolors="black"
            # edge_color = [color for _, _, color in crime_network.edges.data("color")]
        )
        ax[i,1].set_title(
            crime_network.graph["name"] + ": K-means Clustering", 
            fontsize=30
        )

        # Get degree Distribution
        dg = [v for k,v in crime_network.degree()]
        g = sns.histplot(dg, ax=ax[i,2], bins=len(np.unique(dg)))
        g.set(xlabel = "Node Degree", ylabel = "Frequency")
        ax[i,2].set_title("Degree Histogram")
        ax[i,2].axvline(x=np.mean(dg), color='r', linestyle=':')    

        sns.despine(bottom = True, left = True)

        #### Network metadata

        # Eigenvector Centralities
        ec = np.array(list(nx.eigenvector_centrality_numpy(crime_network).values()))
        ec_mean, ec_max, ec_min = np.mean(ec).round(3), np.max(ec).round(3), np.min(ec).round(3)
        # Degrees
        dg = [v for k,v in crime_network.degree()]
        dg_mean, dg_max, dg_min = np.mean(dg).round(1), np.max(dg), np.min(dg)
        # Diameter
        dm = nx.algorithms.distance_measures.diameter(crime_network)
        # Triangles
        A = nx.linalg.graphmatrix.adjacency_matrix(crime_network).todense()
        tr = int(np.trace(np.matmul(A,np.matmul(A.T,A)))/6)
        # Average Clustering
        ac = round(nx.algorithms.cluster.average_clustering(crime_network),3)
        # Nodes and edges
        ns = crime_network.number_of_nodes()
        es = crime_network.number_of_edges()

        res = [
            crime_network.graph["name"],
            ns,
            es,
            dg_mean,
            dg_max,
            dg_min,
            dm,
            tr,
            ac,
            ec_mean, 
            ec_max,
            ec_min
        ]

        results.append(res)

    # ax.set_title(crime_network.graph["name"], fontsize=30)
    print("Saving")
    fig.savefig(basedir / f'batch{set}.png', bbox_inches='tight', dpi=600)



# %%
import pandas as pd
df = pd.DataFrame(
    results, 
    columns=[
        "name", 
        "nodes", 
        "edges", 
        "degree_mean", 
        "degree_max", 
        "degree_min", 
        "diameter",
        "triangles",
        "average_clustering",
        "eig_cent_mean",
        "eig_cent_max",
        "eig_cent_min"
    ]
)
# %%
