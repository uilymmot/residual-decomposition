import numpy as np
import pandas as pd
import copy
import math

class ResidualDecompositionSymmetric():
    def __val_func__(self, x, y, e_x, e_y, model): 
        model.fit(x, y)
        return model.predict(e_x) - e_y
    
    def __init__(self):
        pass
        
    def fit(self, X1, Y1, model, iterations=100):
        self.model = model
        self.X1 = X1
        self.Y1 = Y1
        self.niter = iterations
        self.N = self.X1.shape[0]
        self.k = self.X1.shape[1]
        
        self.__main()
        
    def __main(self):
        # Define our running set to keep track of phi_i for all i
        self.phi = np.zeros((self.N, self.N)) 

        indices = np.arange(0, self.N)

        for j in range(self.niter):

            # Generate a permutation pi from our sample of users
            permutation = np.random.permutation(indices) 

            # Base case, compute the utility $U$ of the first user 
            subset = permutation[:1]

            Xs = self.X1[subset]
            Ys = self.Y1[subset]

            p_i_pi = 0
            p_i_pi_cup_i = self.__val_func__(Xs, Ys, self.X1, self.Y1, self.model)

            self.phi[subset[0]] += p_i_pi_cup_i - p_i_pi

            # Base case, compute the utility $U$ of the rest of the users
            for i in range(1,self.N):

                subset = permutation[:i+1]

                Xs = self.X1[subset]
                Ys = self.Y1[subset]

                p_i_pi = p_i_pi_cup_i
                p_i_pi_cup_i = self.__val_func__(Xs, Ys, self.X1, self.Y1, self.model)

                self.phi[subset[i]] += p_i_pi_cup_i - p_i_pi

            if j % 50 == 0:
                print(j)

        # Take the mean over phi to obtain our composition values
        self.phi /= self.niter
        self.composition = self.phi
        
    def get_composition(self):
        return self.composition
    
    def get_contribution(self):
        summed_composition_residuals = np.sum(self.composition, axis=0)
        return ((self.composition.T * -np.sign(summed_composition_residuals))).T
    
class ResidualDecompositionSymmetricStopping():
    def __val_func__(self, x, y, e_x, e_y, model): 
        model.fit(x, y)
        return model.predict(e_x) - e_y
    
    def __init__(self):
        pass
        
    def fit(self, X1, Y1, model, iterations=10000):
        self.model = model
        self.X1 = X1
        self.Y1 = Y1
        self.niter = iterations
        self.N = self.X1.shape[0]
        self.k = self.X1.shape[1]
        
        self.__main()
        
    def __main(self):
        # Define our running set to keep track of phi_i for all i
        self.phi = np.zeros((self.N, self.N)) 

        indices = np.arange(0, self.N)
        phi_previous = copy.deepcopy(self.phi)
        
        self.n_count = 0
        for j in range(1, self.niter+1):

            # Generate a permutation pi from our sample of users
            permutation = np.random.permutation(indices) 

            # Base case, compute the utility $U$ of the first user 
            subset = permutation[:1]

            Xs = self.X1[subset]
            Ys = self.Y1[subset]

            p_i_pi = 0
            p_i_pi_cup_i = self.__val_func__(Xs, Ys, self.X1, self.Y1, self.model)

            self.phi[subset[0]] += p_i_pi_cup_i - p_i_pi

            # Base case, compute the utility $U$ of the rest of the users
            for i in range(1,self.N):

                subset = permutation[:i+1]

                Xs = self.X1[subset]
                Ys = self.Y1[subset]

                p_i_pi = p_i_pi_cup_i
                p_i_pi_cup_i = self.__val_func__(Xs, Ys, self.X1, self.Y1, self.model)

                self.phi[subset[i]] += p_i_pi_cup_i - p_i_pi
            
            self.n_count += 1
            if j % 100 == 0:
                phi_sum_previous = np.sum(np.abs(phi_previous)) / j
                phi_sum_current = np.sum(np.abs(self.phi)) / j
                if (phi_sum_previous > 0.95 * phi_sum_current) and (phi_sum_previous < 1.05 * phi_sum_current):
                    break
                phi_previous = copy.deepcopy(self.phi)
            
        # Take the mean over phi to obtain our composition values
        self.phi /= self.n_count
        self.composition = self.phi
        
    def get_composition(self):
        return self.composition
    
    def get_contribution(self):
        summed_composition_residuals = np.sum(self.composition, axis=0)
        return ((self.composition.T * -np.sign(summed_composition_residuals))).T
        
class ResidualDecompositionAsymmetric():
    def __val_func__(self, X1, X2, Y1, Y2, model): 
        model.fit(X1, Y1)
        return model.predict(X2) - Y2
    
    def __init__(self):
        pass
        
    def fit(self, X_tr, X_te, Y_tr, Y_te, model, iterations):
        self.X1 = X_tr
        self.Y1 = Y_tr
        self.X2 = X_te
        self.Y2 = Y_te
        self.model = model
        self.N = self.X1.shape[0]
        self.niter = iterations
        
        self.main()
        
    def main(self):
        # Define our running set to keep track of phi_i for all i
        self.phi = np.zeros((self.X1.shape[0], self.X2.shape[0])) 

        indices = np.arange(0, self.N)

        for j in range(self.niter):

            # Generate a permutation pi from our sample of users
            permutation = np.random.permutation(indices) 

            # Base case, compute the utility $U$ of the first user 
            subset = permutation[:1]

            Xs = self.X1[subset]
            Ys = self.Y1[subset]

            p_i_pi = 0
            p_i_pi_cup_i = self.__val_func__(Xs, self.X2, Ys, self.Y2, self.model)

            self.phi[subset[0]] += p_i_pi_cup_i - p_i_pi

            # Base case, compute the utility $U$ of the rest of the users
            for i in range(1,self.N):

                subset = permutation[:i+1]

                Xs = self.X1[subset]
                Ys = self.Y1[subset]

                p_i_pi = p_i_pi_cup_i
                p_i_pi_cup_i = self.__val_func__(Xs, self.X2, Ys, self.Y2, self.model)

                self.phi[subset[i]] += p_i_pi_cup_i - p_i_pi

            if j % 50 == 0:
                print(j)

        # Take the mean over phi to obtain our random variable
        self.phi /= self.niter
        self.composition = self.phi

    def get_composition(self):
        return self.composition
    
    def get_contribution(self):
        summed_composition_residuals = np.sum(self.composition, axis=0)
        return ((self.composition * -np.sign(summed_composition_residuals)))
    
    
class Data_Shapley:

    def __init__(self):
        pass
    
    def fit(self, data, background_data, labels, background_labels, metric, model, iterations=100):
        
        self.X1 = data
        self.X2 = background_data
        self.Y1 = labels
        self.Y2 = background_labels
        self.n, self.k = self.X1.shape
        self.iterations = iterations
        self.metric = metric
        self.model = model
        self.val_track = np.zeros(self.n)
        
        V_S_av = []
        shapley_base = np.zeros(self.n)

        for it in range(1, self.iterations):
            permu = np.random.permutation(np.arange(0, self.n))

            V_S = np.zeros(self.n)

            score_prev = 0
            for i in range(1, self.n-1):
                current_permutation = permu[:i+1]
                X_temp = self.X1[current_permutation,:]
                Y_temp = self.Y1[current_permutation]
                
                self.model.fit(X_temp, Y_temp)
                score = self.metric(self.model.predict(self.X2), self.Y2)

                diff = score - score_prev
                score_prev = score

                shapley_base[permu[i]] += diff
                self.val_track[permu[i]] += 1

            if (it % 25 == 0):
                print("iteration ", it)
            V_S_av.append(V_S)
            
        self.shap_raw = shapley_base
        self.shap_vals = self.shap_raw / self.val_track
        

class ResidualDecompositionSymmetricInfluenceFunctionLinear:
    def __LR_Augment(self, model, X):
        Xa = np.ones((X.shape[0],X.shape[1]+1))
        Xa[:,1:] = X
        A = np.ones(X.shape[1]+1)
        A[0] = model.intercept_
        A[1:] = model.coef_
        return A, Xa
    
    def __LR_HINV(self, X):
        return np.linalg.inv((X.T @ X) + 1e-8 * np.identity(X.shape[1]))
    
    def __Ridge_HINV(self, X, alpha):
        return np.linalg.inv((X.T @ X) + (alpha * np.identity(X.shape[1])))
    
    def __init__(self, X, Y, model):
        self.phi = []
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.k = X.shape[1]
        self.mmatrix = np.zeros((self.n, self.n))
        
        self.residuals_matrix = np.zeros((self.n, self.n))
        self.K_temp_residuals_matrix = np.zeros((self.n, self.n))
        self.counting_matrix = np.zeros(self.n)
        
        self.model = model
        self.rpermlist = []
        
    def step(self, r_perm):
        # Get the subset of data and fit model
        X_sub = self.X[r_perm,:]
        Y_sub = self.Y[r_perm]
        self.model.fit(X_sub, Y_sub)
        r_base = self.model.predict(self.X) - self.Y
        
        # Get the augmented data and model parameters
        aW, aX = self.__LR_Augment(self.model, self.X)
        aX_sub = aX[r_perm,:]
        
        sn = aX_sub.shape[0]
        sk = aX_sub.shape[1]
        
        # Compute influence of this subset of instances

        # Invert the Hessian
        # Replace with appropriate functions for given model type
        
        H_inv = self.__Ridge_HINV(aX_sub, self.model.alpha)# (np.random.rand(*aX_sub.shape) * 1e-32))
        # H_inv = self.__LR_HINV(aX_sub)

        # Compute the first derivative of the loss over all z's of interest
        z_all = np.multiply(np.repeat(((aX_sub @ aW) - Y_sub), sk).reshape(aX_sub.shape), aX_sub)

        # Effect of all z's on the parameters of the LR model
        I_all = (H_inv @ z_all.T).T
        
        # set of parameters of model, duplicated |z| times
        A_new_matrix = np.repeat(aW, sn).reshape(sk, sn).T
        A_new_matrix -= I_all

        # compute the residuals of the LR model using the new parameters
        c_residuals = r_base - ((A_new_matrix @ aX.T) - self.Y)
        
        return c_residuals
            
    def stepUNorm(self, i):
        kv = np.random.randint(1, self.n, i)
        # kv = np.random.choice([1,self.n], i)
        
        set_size_counter = 1
        for size_k in kv:
            k_permutation = np.random.choice(self.n, size_k, replace=False)
            working_residuals = self.step(k_permutation)
            self.counting_matrix[k_permutation] += 1
            self.residuals_matrix[k_permutation] += working_residuals
            
            set_size_counter += 1
            
        self.residuals_matrix /= self.counting_matrix
        
        self.residuals_matrix /= 2
        
    def All_S_Influence(self, i):
        kv = np.random.randint(1, self.n, i)
        kv = np.sort(kv)
        temp_residuals_matrix = np.zeros((self.n, self.n))
        
        number_of_current_k = 1
        count_of_total_k = 1
        current_k = -1
        
        for size_k in kv:
            if (size_k != current_k):
                new_factor = 1 / count_of_total_k
                old_factor = (count_of_total_k - 1) / count_of_total_k
                self.residuals_matrix = (self.residuals_matrix * old_factor) + (temp_residuals_matrix * new_factor)
                temp_residuals_matrix = np.zeros((self.n, self.n))
                number_of_current_k = 1
                count_of_total_k += 1
            
            k_permutation = np.random.choice(self.n, size_k, replace=False)
            working_residuals = self.step(k_permutation)
            
            old_factor = (number_of_current_k - 1) / number_of_current_k
            new_factor = (1 / number_of_current_k)
            temp_residuals_matrix[k_permutation] = (temp_residuals_matrix[k_permutation] * old_factor) + (new_factor * working_residuals)

            number_of_current_k += 1
            current_k = size_k 
        
        self.residuals_matrix /= 2
        
    def Largest_S_Influence(self, i):
        kv = np.random.choice([1, self.n], i)
        # kv = [self.n]
        temp_residuals_matrix = np.zeros((self.n, self.n))

        number_of_current_k = 1
        count_of_total_k = 1
        current_k = -1

        for size_k in kv:
            if (size_k != current_k):
                new_factor = 1 / count_of_total_k
                old_factor = (count_of_total_k - 1) / count_of_total_k
                self.residuals_matrix = (self.residuals_matrix * old_factor) + (temp_residuals_matrix * new_factor)
                temp_residuals_matrix = np.zeros((self.n, self.n))
                number_of_current_k = 1
                count_of_total_k += 1

            k_permutation = np.random.choice(self.n, size_k, replace=False)
            working_residuals = self.step(k_permutation)

            old_factor = (number_of_current_k - 1) / number_of_current_k
            new_factor = (1 / number_of_current_k)
            temp_residuals_matrix[k_permutation] = (temp_residuals_matrix[k_permutation] * old_factor) + (new_factor * working_residuals)

            number_of_current_k += 1
            current_k = size_k 

        self.residuals_matrix /= 2

class residual_shap_WLSN:
    
    def __nCr(self, n,r):
        f = math.factorial
        return f(n) // f(r) // f(n-r)

    def __shapley_kernel(self, M, S):
        return (M-1) / (self.__nCr(M, S) * S * (M - S))
    
    def __init__(self, dX, dY, model, D=0, reg=0.01, override=False, offset=0, onesided=False, rseed=0, \
                 distribution='shapley', weightings='shapley', algorithm='static', max_iter=2000):
        '''

        Parameters
        ----------
        dX : numpy.ndarray
            Input Data X
        dY : numpy.ndarray
            Input Data Y
        model : callable class
            Model to with .fit() and .predict() functions to use dX, dY with
        D : int, optional
            Number of samples/rows to generate our V_D and Z_D matrices with (defaults 2N)
        reg : float, optional
            Regularisation term for the weighted least squares regression
        override : TYPE, optional
            DESCRIPTION. The default is False.
        offset : TYPE, optional
            DESCRIPTION. The default is 0.
        onesided : TYPE, optional
            DESCRIPTION. The default is False.
        rseed : TYPE, optional
            DESCRIPTION. The default is 0.
        distribution : TYPE, optional
            DESCRIPTION. The default is 'shapley'.
        weightings : TYPE, optional
            DESCRIPTION. The default is 'shapley'.
        algorithm : TYPE, optional
            DESCRIPTION. The default is 'static'.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 2000.

        Returns
        -------
        None.

        '''
        self.rseed = rseed
        np.random.seed(self.rseed)
        
        self.X = dX
        self.Y = dY
        self.regulariser = reg
        self.model = model
        self.N1 = self.X.shape[0]
        self.offset = offset
        self.onesided = onesided
        self.override = override
        self.distribution = distribution
        self.weightings = weightings
        self.algorithm = algorithm
        # self.step_size = step_size
        self.max_iter = max_iter
        
        if (D == 0):
            self.D_samples = self.N1 * 2
        else: 
            self.D_samples = D
        
        M = self.N1
        S = 1 + self.offset
        values = []

        while S < M - self.offset:
            skernel = self.__shapley_kernel(M, S)
            values.append(skernel)
            S+=1
            
        values /= np.sum(values)
        self.dis = values
        
        if (self.algorithm == 'static'):
            self.generate_R_D()
            
            self.generate_residual_shap()
        elif (self.algorithm == 'iterative'):
            print("iterative algorithm")
            
            self.iterative_generate_residual()
            
    def iterative_generate_residual(self):
        D = []
        V_D = []
        f_matrix = []
        goodness_history = []
        
        for i in range(1, self.max_iter):
            
            if (self.distribution == 'uniform'):
                v_row = np.random.randint(1, self.N1)
                f_row = self.__gen_Drows(self.N1 - v_row, self.N1)
            elif (self.distribution == 'shapley'):
                print("not implemented")
            
            sset = np.where(f_row == 1)
            
            f_matrix.append(f_row)
            D.append(v_row)
            
            Xs = self.X[sset]
            Ys = self.Y[sset]
            
            self.model.fit(Xs, Ys)
            V_D.append((self.model.predict(self.X) - self.Y).flatten())
            
            Z_D = np.zeros((len(D), self.N1+1))
            Z_D[:,0] = 1
            Z_D[:,1:] = np.array(f_matrix)
            
            W_D = np.zeros((len(D), len(D)))
            counter = 0
            for j in D:
                if (self.weightings == 'shapley'):
                    W_D[counter, counter] = 1 / self.__shapley_kernel(self.N1, j)
                elif (self.weightings == 'uniform'):
                    W_D[counter, counter] = 1 / len(D)
                counter+=1
            
            reg_matrix = (self.regulariser * np.identity(self.N1+1))
            reg_matrix[0,0] = 0
            
            R_D = np.linalg.inv(Z_D.T @ W_D @ Z_D + reg_matrix) @ Z_D.T @ W_D
            
            Svals = R_D @ V_D
            
            self.Svals = Svals
            
            gval = self.check_goodness()
            if (gval > 0.95 and i > self.N1 * 2): #or (len(goodness_history) > 1 and gval == goodness_history[-1])):
                goodness_history.append(gval)
                
                self.V_D = np.array(V_D)
                self.R_D = R_D
                self.W_D = W_D
                self.D = D
                self.goodness_history = goodness_history
                self.tsteps = i
                break
            goodness_history.append(gval)
            
        self.Z_D = Z_D
        self.D = D
        self.V_D = np.array(V_D)
        self.R_D = R_D
        self.W_D = W_D
        self.goodness_history = goodness_history
        self.tsteps = i
                
    def __gen_Drows(self, nzeros, k):
        
        a = np.full(k, 1)
        p = np.random.permutation(np.arange(0, k))
        a[p[:nzeros]] = 0
        return a    
        
    def generate_R_D(self):
        np.random.seed(self.rseed)
        
        if (self.onesided):
            o_probs = np.array_split(self.dis, 2)[-1]
            o_probs /= np.sum(o_probs)
            o_num = np.array_split(np.arange(1+self.offset, self.N1-self.offset), 2)[-1]
            D = np.random.choice(o_num, size = self.D_samples, replace=True, p=o_probs)
        else:
            D = np.random.choice(np.arange(1+self.offset, self.N1-self.offset), size = self.D_samples, replace=True, p=self.dis)

        if (self.override):
            D[:] = self.N1 - self.offset
        
        if (self.distribution != 'shapley'):
            if (self.distribution == 'uniform'):
                D = np.random.randint(1+self.offset, self.N1-self.offset, self.D_samples)
                
        self.D = D
                    
            
        # Generate Z_D matrix for samples 

        Z_D = np.zeros((self.D_samples, self.N1))

        counter = 0
        for i in D:
            Z_D[counter,:] = self.__gen_Drows(self.N1-i, self.N1)
            counter+=1
            
        self.f_matrix = Z_D

        # generate W_D matrix with Shapley kernel weights along the diagonal

        W_D = np.zeros((self.D_samples, self.D_samples))
        counter = 0
        for i in D:
            if (self.weightings == 'shapley'):
                W_D[counter, counter] = self.__shapley_kernel(self.N1, D[counter])
            elif (self.weightings == 'uniform'):
                W_D[counter, counter] = 1 / self.D_samples;
            else:
                print("weighings not defined")
            counter+=1

        reg_matrix = (self.regulariser * np.identity(self.N1))
        reg_matrix[0,0] = 0

        self.R_D = np.linalg.inv(Z_D.T @ W_D @ Z_D + reg_matrix) @ Z_D.T @ W_D
        self.W_D = W_D
        self.Z_D = Z_D
        
    def generate_residual_shap(self):
        V_D = np.zeros((self.D_samples, self.N1))

        for l in range(0, self.D_samples):

            sset = np.where(self.f_matrix[l] == 1)

            Xs = self.X[sset]
            Ys = self.Y[sset]

            self.model.fit(Xs, Ys)

            V_D[l] = (self.model.predict(self.X) - self.Y)
            
        self.V_D = V_D
        
        self.Svals = self.R_D @ V_D
        
    def check_goodness(self):
        self.model.fit(self.X, self.Y.flatten())
        
        A = np.sum(self.Svals, axis=0).reshape(-1,1)
        B = (self.model.predict(self.X) - self.Y).reshape(-1,1)
        
        lr = LinearRegression()
        lr.fit(A, B)
        
        p_residuals = lr.predict(A) - B
        
        self.F_coef_ = lr.coef_
        self.F_intercept_ = lr.intercept_
        self.F_mean = np.mean(np.abs(p_residuals))
        
        return np.abs((np.abs(self.F_coef_) * (1 - np.abs(self.F_intercept_))).flatten()[0])
    
    def __rdis(self, d):
        return np.sqrt(np.sum(d ** 2, axis=1)) * (np.sign(d[:,0]) * np.sign(d[:,1]))
    
    def generate_aggregates(self):
        
        variances = np.array([np.var(self.Svals[1:,:], axis=0), np.var(self.Svals, axis=1)[1:]]).T
        sums = np.array([np.sum(self.Svals, axis=0), np.sum(self.Svals, axis=1)[1:]]).T
        
        self.sAgg = self.__rdis(sums)
        self.vAgg = self.__rdis(variances)
        
        return np.array([self.__rdis(variances), self.__rdis(sums)])