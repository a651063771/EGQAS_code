import wfdb
import numpy as np
from DP import DPCompress
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import MiniBatchKMeans,KMeans
import sklearn.metrics as sm
import networkx as nx
import community as community_louvain
from pykalman import KalmanFilter
#Kalman filter
def Kalman1D(observations, damping=0.5):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

#Compute the feature vector of candidate points
def compute_vector(points_list):
    vectors=[]
    for i in range(1,len(points_list)-1):
        vdist1=points_list[i][1][0]-points_list[i-1][1][0]
        vdist2=points_list[i+1][1][0]-points_list[i][1][0]
        tdist1=((points_list[i][0])-(points_list[i-1][0]))#*(1/360)
        tdist2=((points_list[i+1][0])-(points_list[i][0]))#*(1/360)
        vectors.append([vdist1,vdist2,tdist1,tdist2])

    vectors = np.array(vectors)
    scaler = StandardScaler().fit(vectors)
    vectors = scaler.transform(vectors).tolist()  # 转换X
    vectors_index=[]
    for i in range(len(vectors)):
        vectors[i]=[points_list[i+1][0],vectors[i]]
        vectors_index.append(points_list[i+1][0])
    return vectors,vectors_index
#Construction graph structure (including edge weight aggregation and filtering)
def compute_vector_networkx(vectors):
    nodes = []
    edges = []
    w_list = []
    edges_origin=[]
    for i in range(len(vectors)-1):
        count=0
        nodes.append(vectors[i][0])
        w_temp=[]
        for j in range(i+1,len(vectors)):
            w_sum=np.sum((np.array(vectors[i][1]) - np.array(vectors[j][1])) ** 2)
            if w_sum==0:
                continue
            w= 1/(np.sqrt(np.sum((np.array(vectors[i][1]) - np.array(vectors[j][1])) ** 2)))
            edges_origin.append((vectors[i][0], vectors[j][0], w))
            w=sigmoid(w)
            w_list.append(w)
            w_temp.append((w,count))
            edges.append((vectors[i][0],vectors[j][0],w))
    # Calculate the edge weight threshold
    edges_final=[]
    w_list_label=MiniBatchKMeans(n_clusters=2).fit_predict(np.reshape(w_list,[len(w_list),1]))
    w_list_0=[]
    w_list_1=[]
    for i,w_label in enumerate(w_list_label):
        if w_label==0:
            w_list_0.append(w_list[i])
        else:
            w_list_1.append(w_list[i])
    if np.mean(w_list_1)>np.mean(w_list_0):
        min_index=min(w_list_1)
    else:
        min_index=min(w_list_0)
    for edge in edges:
        if edge[2]>min_index:
            edges_final.append((edge[0],edge[1],edge[2]))
        else:
            edges_final.append((edge[0], edge[1], 0))

    G = nx.Graph()
    G_amc=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges_final)
    G_amc.add_nodes_from(nodes)
    G_amc.add_weighted_edges_from(edges_origin)
    partition,in_degree,degree,links = community_louvain.best_partition(G)
    dendrogram,_,_,_ = community_louvain.generate_dendrogram(G)
    _, in_degree, degree, links = community_louvain.modularity(partition, G)
    partitions = []
    partition_clusters=[]
    for i in set(partition.values()):
        partition_i = []
        partition_cluster=[]
        for j in partition.keys():
            if partition[j] == i:
                partition_i.append(j)
                partition_cluster.append(i)
        partitions.append(partition_i)
        partition_clusters.append(partition_cluster)
    return partitions,partition_clusters,in_degree,degree,links

def compute_AMC(patition_len,s_in,s_out,m):
    m2 = m*2
    amc = (s_in / m2 - (s_out / m2) ** 2)/(patition_len)
    return amc
#Build the first-layer of modified k-means algorithm
def compute_kmeans(partition):
    partition_weight = []
    for i in partition:
        partition_weight.append([len(i)])
    cluster = KMeans(n_clusters=2).fit(partition_weight)
    result = cluster.predict(partition_weight)
    return cluster,partition_weight,result
#Remove abnormal clusters according to the first layer k-means algorithm
def compute_normal_com(cluster,partition_weight,partition,result):

    centroid = cluster.cluster_centers_
    centroid1 = sorted(centroid)
    index=centroid.tolist().index(centroid1[-1][0])
    partition_final = []
    partition_final_index = []
    for i in range(len(partition_weight)):
        if result[i]==index:
            partition_final.append(partition[i])
            partition_final_index.append(i)
    return partition_final,partition_final_index
score_list=[]
#data_paths=#[100,101,102,103,104,105,106,107,108,
data_paths_mitdb = [100,101,102,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,\
            200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234]
for path in data_paths_mitdb:
    #record = wfdb.rdrecord('./mgh003/mgh003', physical=False, channels=[3, ])
    record = wfdb.rdrecord('./mitdb/'+str(path), sampfrom=0,sampto=650000, physical=False, channels=[0, ])
    #record = wfdb.rdrecord('./wrist/s5_run', physical=False, channels=[4, ])
    #record = wfdb.rdrecord('./als1/als1', physical=False, channels=[0, ])
    ventricular_signal = record.d_signal
    X_train = np.array(ventricular_signal)
    ventricular_signal_origin=ventricular_signal
    ventricular_signal= np.reshape(ventricular_signal, [len(ventricular_signal), 1])
    scaler = StandardScaler().fit(ventricular_signal)
    #scaler = MinMaxScaler().fit(ventricular_signal)
    ventricular_signal = scaler.transform(ventricular_signal)  # 转换X

    signal_annotation = wfdb.rdann('./mitdb/'+str(path), "atr",summarize_labels=True,sampfrom=0,sampto=100000)
    if path in [104, 106, 108, 200, 201, 205, 207, 208, 209, 215, 221, 222, 231, 232]:
        print("filter")
        ventricular_signal = Kalman1D(ventricular_signal)
        ventricular_signal = np.reshape(ventricular_signal, [len(ventricular_signal), 1])
    Pointlists=[]
    Pointlists_origin=[]
    for index,value in enumerate(ventricular_signal):
        Pointlists.append([index*(1/360),value])
    for index,value in enumerate(ventricular_signal_origin):
        Pointlists_origin.append([index*(1/360),value])
    #use DP algorithm to obtain candidate points
    dp=DPCompress(Pointlists,0.3)
    points = dp.runDP(Pointlists,0.3)
    #Using EGQAS algorithm to realize periodic point detection
    vectors,vectors_index=compute_vector(points)
    partition,partition_clusters,s_in,s_out,s_m=compute_vector_networkx(vectors)
    cluster,partition_weight,result=compute_kmeans(partition)
    partition_final,partition_final_index=compute_normal_com(cluster,partition_weight,partition,result)
    partition_value=[]
    vectors_final = []
    index_list=np.array(Pointlists_origin)[:, 0].tolist()
    for i in range(len(partition_final)):
        for j in partition_final[i]:
            index=index_list.index(j)
            partition_value.append([Pointlists_origin[index][0], Pointlists_origin[index][1]])
            vectors_final.append(vectors[vectors_index.index(j)][1])
    cluster_result_final=[]
    cluster_result_finals=[]
    for i in range(len(partition_final_index)):
        for _ in range(len(partition[partition_final_index[i]])):
            cluster_result_finals.append(partition_final_index[i])
    sil_values=sm.silhouette_samples(vectors_final, cluster_result_finals, metric='euclidean')
    sil_score=sm.silhouette_score(vectors_final, cluster_result_finals, metric='euclidean')
    cluster_list=[]
    cluster_values=[]
    cluster_values_mean=[]
    cluster_set=set(cluster_result_finals)
    for i in cluster_set:
        cluster_values_sub=[]
        cluster_labels_sub=[]
        for j in range(len(cluster_result_finals)):
            if cluster_result_finals[j]== i:
                cluster_values_sub.append(sil_values[j])
        cluster_values.append(cluster_values_sub)
    for i in range(len(cluster_values)):
        cluster_values_mean.append(np.mean(cluster_values[i]))

    score_list.append(sm.silhouette_score(vectors_final, cluster_result_finals, metric='euclidean'))
    AMC_value = []
    AMC_index = []
    period_best=[]
    period=[]
    period_std=[]
    for part in partition_final:
        period_sub=[]
        part=sorted(part)
        for i in range(1,len(part)):
            period_sub.append(part[i]-part[i-1])
        period_std.append(np.std(period_sub))
        period.append(period_sub)
    #second-layer modified kmeans
    period_std=np.reshape(period_std,[len(period_std),1])
    std_result=KMeans(n_clusters=2).fit_predict(period_std)
    std_partition=[]
    period_std_cluster_mean=[]
    for i in set(std_result):
        std_partition_sub=[]
        period_std_cluster_sub=[]
        for j in range(len(std_result)):
            if std_result[j]==i:
                std_partition_sub.append(partition_final_index[j])
                period_std_cluster_sub.append(period_std[j])
        std_partition.append(std_partition_sub)
        period_std_cluster_mean.append(np.mean(period_std_cluster_sub))
    std_index=period_std_cluster_mean.index(min(period_std_cluster_mean))
    best_std_index=std_partition[std_index]
    cluster_values_mean_final=[]
    for i in best_std_index:
        AMC_value.append((compute_AMC(partition_weight[i][0], s_in[i], s_out[i], s_m),i))
    # Cluster evaluation after removing outliers and incomplete split points
    vectors_final_sec = []
    index_list = np.array(Pointlists_origin)[:, 0].tolist()

    for i in range(len(partition_final)):
        if partition_final_index[i] in best_std_index:
            for j in partition_final[i]:
                index = index_list.index(j)
                partition_value.append([Pointlists_origin[index][0], Pointlists_origin[index][1]])
                vectors_final_sec.append(vectors[vectors_index.index(j)][1])

    cluster_result_finals_sec = []
    for i in range(len(partition_final_index)):
        if partition_final_index[i] in best_std_index:

            for _ in range(len(partition[partition_final_index[i]])):
                cluster_result_finals_sec.append(partition_final_index[i])
    if len(best_std_index)>1:
        print("MSV value", sm.silhouette_score(vectors_final_sec, cluster_result_finals_sec, metric='euclidean'))

