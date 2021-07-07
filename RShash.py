import numpy as np

class RShash():
    """ score datapoints as outliers 
    
    Implementation of rs-hash algorithm for outlier scoring according to 
    Sathe and Aggarwal "Subspace Outlier Detection in Linear Time with Randomized Hashing", DOI: 10.1109/ICDM.2016.0057

    """
    def __init__(self,n_hashes=4,n_samples=1000,n_runs=300,seed=None):
        self.n_hashes = n_hashes
        self.n_samples = n_samples
        self.n_runs = n_runs
        np.random.seed(seed)
        self.seeds_per_run = np.random.randint(low=0,high=np.power(2,30),size=n_runs)


    def score(self,data):
        n_data = data.shape[0]
        # initilise array to store scores of all runs (for insight into outliers)
        scores_all_runs = np.zeros((self.n_runs,n_data)) 
        # alternatively only get the average score, updated in each run
        # avg_score = 0  
        for k,seed in enumerate(self.seeds_per_run):
            # call function to put all data points into a random defined grid using the seed passed to the function
            y_bar, sample_obs, dim_keep = self._put_data_in_grid(data,seed)
            # in order to apply n_hash differnt hashings, append a number from 0 to n_hash-1 to each element of y_bar
            data_arrays_to_hash = np.hstack( (np.tile(y_bar, (self.n_hashes,1)),
                                                np.reshape( np.repeat(range(self.n_hashes),n_data),
                                                           (self.n_hashes*n_data,1))
                                             )
                             )
            # the counts in the hash table are done for the observation sample only, select these elements
            sample_to_hash = data_arrays_to_hash[np.concatenate([sample_obs+k*n_data for k in range(self.n_hashes)]),]
            # and then cerate hashtable with counts
            hashtab = {}
            for arr in sample_to_hash:
                hashtab[arr.data.tobytes()] = hashtab.get(arr.data.tobytes(), 0) + 1
            # then assign these counts to the whole population
            all_counts  =  np.array( [hashtab.get(data_array.data.tobytes(), 0) for data_array in data_arrays_to_hash] )
            # get score, i.e., take the minimum of the counts in all n_hash hash tables
            # and add +1 to the counts for out of sample points and then take the log
            score = np.log(
                              np.reshape( all_counts, (self.n_hashes,n_data) ).min(axis=0)
                            + (1 - np.isin(np.array(range(n_data)), sample_obs, assume_unique=True))
                    )
            # write this score into results array
            scores_all_runs[k,:] = score
            # below line of not all runs are stored and average updated in each run
            # avg_score  = avg_score*(k/(k+1)) + score/(k+1) 

        return scores_all_runs.mean(axis=0)


    # put each data point in a random defined grid
    def _put_data_in_grid(self,data,seed):
        n_data = data.shape[0]
        n_dim = data.shape[1]
        np.random.seed(seed)
        # sample locality parameter f (step 1 in paper)
        f = np.random.uniform( np.power(self.n_samples,-0.5) , 1-np.power(self.n_samples,-0.5))
        assert(f > np.power(self.n_samples,-0.5) )
        # get r, number of dimensions to use in this hash run
        log_of_s = np.log( self.n_samples ) / np.log( np.maximum(2, (1/f) ) )
        r = np.random.randint( low = np.round(1+0.5*log_of_s), high = np.ceil(log_of_s) , size = 1)
        r = np.minimum(r,n_dim)
        # get sample of dimensions to use
        sample_dims = np.random.choice( range(n_dim), size=np.minimum(r,n_dim), replace=False)
        # and sample of observations to use
        sample_obs = np.random.choice( range(1,n_data), size=self.n_samples, replace=False)
        # get min and max overservation sample in each dimension
        dim_min = data[sample_obs].min(axis=0)
        dim_max = data[sample_obs].max(axis=0)
        # drop from sampled dimensions those that are constant over the observation sample
        dim_keep = np.intersect1d( np.where(dim_min < dim_max)[0], sample_dims )
        # from obs to y_bar and to hash dict
        # linear affine transformmation of the observation to y_bar (notation from paper), i.e.,
        # first step, so that [0,1] is the range in each kept dimension
        # then scaled by 1/f and shifted with a random number from [0,1] in each dimension
        y_bar = np.floor(
                  ( (np.take(data, dim_keep, axis =1) - dim_min[dim_keep])
                   /(dim_max[dim_keep] - dim_min[dim_keep])
                  )/f
                   + np.random.rand(dim_keep.shape[0],)
                )
        return y_bar, sample_obs, dim_keep
