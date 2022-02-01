#%%

#Import all relevant libraries. 
#Installation instructions for graph_tool : https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions
import graph_tool as gt
from graph_tool.topology import extract_largest_component, shortest_distance
from graph_tool.stats import distance_histogram
from scipy.io import loadmat
import os, matplotlib, pickle, time
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# %%
#Define a class to handle all of the networks we will import
class FBNetwork(gt.Graph):
    def __init__(self, **kwg):
        #Inherit from the graph_tool Graph class. kwgs is used to allow us to specify an undirected graph
        super().__init__(**kwg)
        #Create class variables to store the vertex properties
        self.v_status = None
        self.v_flag = None
        self.v_gender = None
        self.v_major = None
        self.v_minor = None
        self.v_dormyear = None
        self.v_highschool = None
        #Class variable to store the giant component
        self.gc = None

    #This method is called to load a .mat file
    def load_mat(self, filename):
        #Load the .mat file using scipy's loadmat which produces a dictionary
        mat = loadmat(filename)
        #We add the adjaceny matrix from the dictionary. Calling .nonzero() returns two rows that match the indices of all non-zero entries in the matrix which are all
        #existing edges in the graph forming a 2xm array. We transpose it to fit the format of having an array of m tuples (mx2) and then simply add the edge list
        #to our graph (self)
        self.add_edge_list(np.transpose(mat["A"].nonzero()))

        #Here we load all the node propoerties into our graph, in case we wnat to use them for analysis. Not necessary, but housekeeping.
        node_properties = mat["local_info"]
        
        self.v_status = self.new_vertex_property("int", vals=node_properties[:,0])
        self.v_flag = self.new_vertex_property("int", vals=node_properties[:,1])
        self.v_gender = self.new_vertex_property("int", vals=node_properties[:,2])
        self.v_major = self.new_vertex_property("int", vals=node_properties[:,3])
        self.v_minor = self.new_vertex_property("int", vals=node_properties[:,4])
        self.v_dorm_year = self.new_vertex_property("int", vals=node_properties[:,5])
        self.v_highschool = self.new_vertex_property("int", vals=node_properties[:,6])
    
    #Simple method that we can call to extract the largest component and save it as a class object
    def get_gc(self):
        self.gc = extract_largest_component(self)

#The main component of this exercise. This method creates a FBNetwork class, loads a .mat file. and gets the relevant statistics. Probably shold have been 
#a static method of the class. 
def get_stats(filepath):
    '''
    For a facebook network file, loads and returns the diameter of largest component, network size
    average geodesic distance of largest component, and size of largest component
    '''
    #Instnatiate an instance of the class and creates an undirected graph
    fb_network = FBNetwork(directed=False)

    #Loads the .mat file
    fb_network.load_mat(filepath)
    
    #Extracts the largest component of the graph
    fb_network.get_gc()
    
    #Calling distance_histogram from graph_tool on the giant component (which is now a graph_view) returns an array with two rows:
    #The first row contains the counts of the different geodesic distances within the graph. 
    #The second row contains the bin values of the geodesic distances
    dist_hist = distance_histogram(fb_network.gc)

    #We store the first row as counts and the second row as the distances. We have to eliminate the last value of the distances because they are
    #histogram bins that are lower bound inclusive and upper bound exclusive
    counts, dists = dist_hist[0], dist_hist[1][:-1]
    
    #Call the max function to get the diameter of the largest component
    l_max = dists.max()

    #We dot product the counts and distances and divide by the total counts to get the mean geodesic distance
    l_mean = np.dot(counts, dists)/counts.sum()

    #Return the stats
    return l_max, fb_network.num_vertices(), l_mean, fb_network.gc.num_vertices()

#Function the read all the files in the folder and return all the ones that end in .mat
#Returns the list of files
def filter_files(filepath, files_read):
    #Compile a regex to match the .mat files
    r = re.compile(".*mat")

    #Filter the files and return the list
    filtered_files = list(filter(r.match, os.listdir(filepath)))
    files = [file for file in filtered_files if file not in files_read]
    return files


#Main run of the exercise
def main(filepath):
    #Try to  load a tmp pickle file where we save the results in between networks in case of crashing or having to go to class.
    try:
        with open("fb_networks_tmp2.pkl", "rb") as infile:
            files_read, diameter, network_size, mean_geodesic, component_size, computation_time = pickle.load(infile)
        print("Previously saved network data loaded. Proceeding to analyze more networks...")

    #If there was no previous data we start from the beginning
    except FileNotFoundError:
        print("No saved network data, starting from scratch.")

        #Empty lists to store the data we need
        files_read = []
        diameter = []
        network_size = []
        mean_geodesic = []
        component_size = []
        computation_time = []
    
    #Filter the files that already have been analyzed out
    remaining_files = filter_files(filepath, files_read)

    #Main loop
    for file in tqdm(remaining_files, desc = "Parsing networks", colour ="blue", position =1):

        #Record start time to know how long it takes
        start_time = time.time()

        #use our get stats function to get all the necessary statistics
        l_max, n_network, l_mean, n_component = get_stats(os.path.join(filepath, file))
        
        #Save the statistics
        diameter.append(l_max)
        network_size.append(n_network)
        mean_geodesic.append(l_mean)
        component_size.append(n_component)
        files_read.append(file)

        #Record the time taken
        elapsed = time.time() - start_time
        computation_time.append(elapsed)

        #Save to file
        with open("fb_networks_tmp2.pkl", "wb") as outfile:
            pickle.dump((files_read, diameter, network_size, mean_geodesic, component_size, computation_time), outfile)
        print(f"{file.strip('.mat')} network saved in {round(elapsed, 2)} seconds for {n_component} nodes!")
    return files_read, diameter, network_size, mean_geodesic, component_size, computation_time

#%%
#Running the main loop

filepath = os.path.join("Facebook 100", "facebook100")
assert os.path.exists(filepath), f"Need to extract facebook data into current directory. Make sure directory path matches. {filepath}"
files_read, diameter, network_size, mean_geodesic, component_size, computation_time, = main(filepath)

#%%

#Load our files for plotting
with open("fb_networks_tmp2.pkl", "rb") as infile:
            files_read, diameter, network_size, mean_geodesic, component_size, computation_time = pickle.load(infile)

#Set style and savepath for the figures
matplotlib.style.use("seaborn-bright")
savepath = "figs"
os.makedirs(savepath, exist_ok=True)

#Function to predict and plot the log fitted trend line
def plot_logtrend(x, y, ax):
    #Take the log of your x values
    logx = np.log(np.array(x))

    #Fit a 1 dimensional polynomial to your line (loglinear now)
    z = np.polyfit(logx, y, 1)

    #Creates the function 
    p = np.poly1d(z)

    #Plot to your axes with a dashed red line
    ax.plot(sorted(x), p(np.sort(logx)), label ="Log Fit", linestyle="dashed", color="red")

#Plotting the diameter plot
diameter_fig, diameter_ax = plt.subplots()
# diameter_ax.scatter(network_size, diameter)
sns.stripplot(x=diameter, y=network_size, ax=diameter_ax, jitter=0.2)
diameter_fig
diameter_ax.set_ylabel("Network Size (n)")
diameter_ax.set_xlabel("Diameter of Largest Component")
diameter_fig.savefig(os.path.join(savepath, "diameter.png"), facecolor="white", transparent=False)


#Plotting th mean geodesic plot with a log trend line
mean_fig, mean_ax = plt.subplots()
mean_ax.scatter(component_size, mean_geodesic)
mean_ax.set_xlabel("Size of Largest Component")
mean_ax.set_ylabel("Mean Geodesic Distance in Largest Component")
plot_logtrend(component_size, mean_geodesic, mean_ax)
mean_fig.savefig(os.path.join(savepath, "mean.png"), facecolor="white", transparent=False)


#Plotting computation time just to know
time_fig, time_ax = plt.subplots()
time_ax.scatter(component_size, computation_time)
time_ax.set_xlabel("Size of Largest Component")
time_ax.set_ylabel("Computation Time")
plot_logtrend(computation_time, component_size, time_ax)
time_fig.savefig(os.path.join(savepath, "comp_time.png"), facecolor="white", transparent=False)

# %%
