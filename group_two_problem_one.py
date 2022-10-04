import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from matplotlib import pyplot as plt
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import great_circle_distance_matrix


data_file = "FieldActivity_log.csv"

# First, we read the .csv file that comes with this .py file
data = pd.read_csv(data_file)

# All the data has more than just a latitude and a longitude on it; We only want those two parts
addresses = data[['Latitude For Service Point', 'Longitude For Service Point']]

# Convert the list to an array for the constrained cluster library to work with.
addressesArray = np.array(addresses)


# Now we split the data up into 15 clusters. Each cluster must have at least 178 stuff in it and can have no more than 180 (for now).
# The reason KMeans Constrained is being used instead of normal KMeans is because there would be massive disparities between 
# some cluster densities that would be unfeasible to try to work out for this hackathon. 
# It's much simpler to have every cluster be roughly the same size.
constrainedCluster = KMeansConstrained(
    n_clusters=15,
    size_min=178,
    size_max=180,
    random_state=0
)
constrainedCluster.fit_predict(addressesArray)

# And we store the cluster information.
labels = constrainedCluster.labels_


# Now we add a new column to the data frame, and we keep that cluster information in that column.
addresses['cluster'] = labels
_clusters = addresses.groupby('cluster').count()


# Euclidean distance calculator. That's the distance between point A and point B in 2-dimensional space. 
# It's only here so that one other action works correctly.
def euclid_calc(point_a, point_b):

    # Since points A and B are not numpy arrays and the numpy normal vector function seems to only work on numpy arrays, points A and B must now become numpy arrays
    working_point_a = np.array((point_a[0], point_a[1]))
    working_point_b = np.array((point_b[0], point_b[1]))

    # Gives back the normal vector between the two points A and B. That's the distance between them, in this case.
    distance = np.linalg.norm(working_point_b - working_point_a)
    # This doesn't return point A because we don't need it.
    return distance, point_b

# Wasted my time typing these, but I won't delete them.
gas_station_one = [30.3700831, -81.6976878, 29]
gas_station_two = [30.4432641, -81.6101161, 33]
gas_station_three =[30.2752551, -81.5621375, 28]
gas_station_four = [30.537959, -81.725028, 31]
gas_station_five = [30.3254941, -81.6508163, 32]
gas_station_six = [30.2042179, -81.581387, 35]
gas_station_seven = [30.3382235, -81.7096026, 30]
gas_station_eight = [30.3119702, -81.5260939, 25]
gas_station_nine = [30.26992, -81.759729, 27]


# The list of coordinates of the gas stations that the trucks have to stop to.
gas_stations_list = [[30.3700831, -81.6976878], [30.4432641, -81.6101161], [30.2752551, -81.5621375], [30.537959, -81.725028], [30.3254941, -81.6508163], [30.2042179, -81.581387], [30.3382235, -81.7096026], [30.3119702, -81.5260939], [30.26992, -81.759729]]


# This function is a travelling salesman problem. 
# The actual algorithm is handled by a tsp library; Would not have been able to write a tsp solution from scratch in time.
# This function will be run 15 times. Once for each cluster.
def tsp_for_one_clust(cluster):


    # Add the depot as a point. Since the traveling salesman is cyclical, any point can be the depot.
    # This will appear last in all the lists. Good thing this is a tsp, otherwise that might have been a problem.
    cluster.loc[-1] = [30.3357486, -81.7568817, 15]

    # A list of indexes for each cluster will come in handy. 
    # The list of points was indexed before it was clustered, so none of the clusters have 
    # an index starting from 0 and going straight up to 179/181. 
    # Therefore, keeping track of what indexes each cluster has is good for us.
    index_list = cluster.index

    # This is the process for putting in a gas station that the truck must stop at. 
    # The station for each cluster will be the closest station (in Euclidean distance) to a random point in the cluster.
    # Pick a random index from the cluster
    random_index = random.choice(index_list)

    # The point will be the object that exists at that index. It will be more than just latitude and longitude though, so we'll take that extra thing out.
    random_point_object = cluster.loc[random_index]
    random_point = [random_point_object['Latitude For Service Point'], random_point_object['Longitude For Service Point']]

    # The default gas station to put in the cluster is the first one.
    shortest_distance, closest_station = euclid_calc(random_point, gas_stations_list[0])
    # print('Setting the default shortest distance to: ')
    # print(shortest_distance)
    # print('and that is between the random point and the closest gas station, with coordinates at: ')
    # print(closest_station)

    # Iterate through the list of gas stations... If you come across any closer ones, then reset the closest station and shortest distance to that one.
    for station in gas_stations_list:
        current_distance, current_station = euclid_calc(random_point, station)
        if(current_distance < shortest_distance):
            # print('New shortest distance! It is: ')
            # print(current_distance)
            # print('And it is at the station located at: ')
            # print(current_station)
            shortest_distance = current_distance
            closest_station = current_station
    
    # And now we have gone through the whole list. Whichever gas station is currently occupying the "closest station" spot is the closest.
    cluster.loc[183] = (closest_station[0], closest_station[1], 28)

    # Now, each cluster has the depot as a start and end point and it has one gas station that the truck must visit at some point during its trip.


    # Sanity check
    print((cluster))
    

    # Uncomment the code below to generate a scatter plot of each cluster. Since the scatter plot is on an XY axis and the points are given with their
    # latitude and longitude, each plot is also a map, showing where each point is relative to the other points. This may be good for visualization.
    # If this code is uncommented, then the program will pause every time a plot is displayed on the screen. 
    # It will resume when the plot is closed (not minimized). This will happen 15 times.

    # plt.scatter(x=cluster['Latitude For Service Point'], y=cluster['Longitude For Service Point'])
    # plt.xlabel('latitude')
    # plt.ylabel('longitude')
    # plt.grid(True)
    # plt.show()


    # Stuff will go in here soon.
    point_list = []

    # And now we iterate through the index list for this cluster. 
    # As we do that, we can isolate individual points in the cluster based on their index.
    for index in index_list:
        cluster_item_right_now = cluster.loc[index]
        
        # and we change that data frame format to just a point with a latitude and a longitude, and then add that point to the list.
        current_point = [cluster_item_right_now['Latitude For Service Point'], cluster_item_right_now['Longitude For Service Point']]
        point_list.append(current_point)

    # Sanity check
    print('Here is the list of points we are working with')
    print(point_list)

    # Converting the points we have into a distance matrix. This is so that we can use the tsp library. 
    # Writing a tsp solution from scratch would probably take too long. to do and debug.
    distance_matrix_source = np.array(point_list)
    distance_matrix = great_circle_distance_matrix(distance_matrix_source)


    # This is that tsp library! It will give back the optimized (as well as possible within a reasonable time) path 
    # and the distance that path covers.

    # The method of finding the shortest path that was applied in this case is a metaheuristic approach.

    # The optimal path is presented as a list of indices. Each index in the list corresponds to a point from the list of points for this cluster.
    # return list_of_points, min_weight
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    print('')
    print('Optimized Path')
    print(permutation)
    print('')
    print('Optimized Distance')
    print(distance)
    print('')


# Iterate through the list of clusters I have. For each one, run the "tsp_for_one_clust" function.
for clust in list(set(addresses['cluster'])):
    clust_i = addresses[addresses['cluster'] == clust]
    #tsp_pts, tsp_weight = tsp_for_one_clust(clust)

    tsp_for_one_clust(clust_i)

# Uncomment the code below to see a scatter plot of the entire system. Each cluster will be a different color.
# This plot will also be a map, but there may be far too many data points too close to each other for it to be a useful map.
# If this code is uncommented, then the plot will not show until everything else has been run. That will take a while.

# plt.scatter(x=addresses['Latitude For Service Point'], y=addresses['Longitude For Service Point'], c=addresses['cluster'])
# plt.xlabel('latitude')
# plt.ylabel('longitude')
# plt.grid(True)
# plt.show()

# This method can be optimized for road distance (it shouldn't make too much of a difference in the optimal order) by applying
# information provided by a Map API, but there was not enough time for that.

# This method can also be updated to account for how much actual time trucks will be on the road for, 
# and separate each truck's journey into two separate trips, but that was another thing that there wasn no time for.
