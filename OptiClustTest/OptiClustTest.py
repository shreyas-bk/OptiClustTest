import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

class Optimal():
    
    """
    To find optimal number of clusters using different optimal clustering algorithms
    *citation*
    *parameters*
    *methods*
    *example*
    """
    
    opti_df = None
    
    def __init__(
        self,
        kmeans_kwargs: dict = None
    ):
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        self.kmeans_kwargs = kmeans_kwargs
    
    def elbow(self,df,upper=15,display=False,visualize=False,function='inertia',method='angle',se_weight=1.5):
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        lower=1
        inertia = []
        K=range(lower,upper)
        for i in K:
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            if function=='inertia':
                inertia.append(cls.inertia_)
            elif function=='distortion':
                inertia.append(sum(np.min(cdist(df, cls.cluster_centers_, 
                          'euclidean'),axis=1)) / df.shape[0])
            else:
                print('function should be "inertia" or "distortion"')
                return -1
        inertia = np.array(inertia)/(np.array(inertia)).max()*(upper-lower)
        slopes = [inertia[0]-inertia[1]]
        for i in range(len(inertia)-1):
            slopes.append(-(inertia[i+1]-inertia[i]))
        angles = []
        for i in range(len(slopes)-1):
            angles.append(np.degrees(np.arctan((slopes[i]-slopes[i+1])/(1+slopes[i]*slopes[i+1]))))
        if display==True:
            plt.plot(K, inertia, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel(function) 
            plt.title('The Elbow Method using '+function) 
            plt.show()
        extra=''
        if method == 'angle':
            optimal = np.array(angles).argmax()+1
            confidence = round(np.array(angles).max()/90*100,2)
            if confidence<=50:
                extra=' with Confidence:'+str(confidence)+'%.'+' Try using elbow_kf, gap_stat_se or other methods'
        elif method == 'lin':
            slopes = [inertia[0]-inertia[1]]
            for i in range(len(inertia)-1):
                slopes.append(-(inertia[i+1]-inertia[i]))
            means = []
            sds = []
            for i in range(len(slopes)):
                means.append(np.mean(np.array(slopes[i:i+3 if i+3<len(slopes) else (i+2 if i+2<len(slopes) else i+1)])))
                sds.append(np.std(np.array(slopes[i:i+3 if i+3<len(slopes) else (i+2 if i+2<len(slopes) else i+1)])))
            diffs = [x[0]-se_weight*x[1]>0 for x in zip(means,sds)]
            optimal = (len(diffs) - list(reversed(diffs)).index(False))
        else:
            print('method should be "angle" or "lin"')
            return -1
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                return optimal, 'Number of columns of the DataFrame should be between 1 and 3 for visualization'
        print('Optimal number of clusters is: ',str(optimal),extra)
        return optimal 
    
    def _visualization(self,df,optimal):
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        cls = KMeans(n_clusters=optimal,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=optimal)
        if len(df.columns) == 1:
            cls_assignment = cls.fit_predict(df)
            col_name = df.columns[0]
            sns.stripplot(data = df,x=['']*len(df),y=col_name,hue=cls_assignment)
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
        elif len(df.columns)==2:
            cls_assignment = cls.fit_predict(df)
            col_name1 = df.columns[0]
            col_name2 = df.columns[1]
            sns.scatterplot(data=df,x=col_name1,y=col_name2,hue=cls_assignment,palette='Set1')
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
        elif len(df.columns)==3:
            cls_assignment = cls.fit_predict(df)
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            col_name1 = df.columns[0]
            col_name2 = df.columns[1]
            col_name3 = df.columns[2]
            ax.scatter3D(xs=df[col_name1],ys=df[col_name2],zs=df[col_name3],c=pd.Series(cls_assignment))
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
        else:
            return 'fail'
        #TODO: add other algos
