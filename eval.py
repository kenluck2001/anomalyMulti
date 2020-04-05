import numpy as np
from anomalymulti import probabilisticMultiEWMA
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def getAARD (y_test, y_pred):
    """
    Both input are numpy array
    """
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    absdiff = np.fabs (y_test - y_pred).reshape ((1, len(y_pred)))
    diff =  np.squeeze(absdiff) / y_test.reshape (( 1, len(y_test)))
    result = (100.0 / len(y_test)) *  np.sum (diff)
    return np.fabs (result)

def bulkUpdate(anom, Z):
    n, m = Z.shape
    for ind in range(n):
        cur = Z[ind].reshape((1,m))
        anom.update(cur)
    cov = anom.getCurrentCovariance()
    incov = anom.getCurrentInvCovariance()
    return cov, incov

def experiment1(X, num_fold = 5):
    output_dict = {}
    n = len(X) # number of samples
    sub_len = n // num_fold
    x_list, cov_loss_list, inv_cov_loss_list = [], [], []

    for i in range(1,num_fold+1):
        start = i * sub_len
        y_train = X[:start] # train
        z_update = X[start:start+sub_len]  # update

        anom = probabilisticMultiEWMA()
        anom.init(y_train)
        cov, incov = bulkUpdate(anom, z_update)
        x_value = X[:start+sub_len]
        orig_cov, orig_incov = anom.getOriginalCovariance(x_value), anom.getOriginalInvCovariance(x_value)

        cov_loss = getAARD (orig_cov, cov)
        inv_cov_loss = getAARD (orig_incov, incov)
        #print ("cov_loss: {}, inv_cov_loss: {}".format(cov_loss, inv_cov_loss))

        x_list.append(start)
        cov_loss_list.append(cov_loss) 
        inv_cov_loss_list.append(inv_cov_loss)

    output_dict["x"] = x_list 
    output_dict["cov_loss"] = cov_loss_list 
    output_dict["inv_cov_loss"] = inv_cov_loss_list
    return output_dict

def experiment2(X, num_fold = 5):
    output_dict = {}
    n = len(X) # number of samples
    sub_len = n // num_fold
    x_list, cov_loss_list, inv_cov_loss_list = [], [], []

    for i in range(1, num_fold+1):
        start = i * sub_len
        y_train = X [:start] # train

        anom = probabilisticMultiEWMA()
        anom.init(y_train)
        remaining = X [start : ] # test

        cov, incov = bulkUpdate(anom, remaining)
        orig_cov, orig_incov = anom.getOriginalCovariance(X), anom.getOriginalInvCovariance(X)
        cov_loss = getAARD (orig_cov, cov)
        inv_cov_loss = getAARD (orig_incov, incov)
        #print ("cov_loss: {}, inv_cov_loss: {}".format(cov_loss, inv_cov_loss))
        x_list.append(start)
        cov_loss_list.append(cov_loss) 
        inv_cov_loss_list.append(inv_cov_loss)

    output_dict["x"] = x_list 
    output_dict["cov_loss"] = cov_loss_list 
    output_dict["inv_cov_loss"] = inv_cov_loss_list
    return output_dict


def draw(data, title, imagefile):
    x = data["x"] 
    y_cov_loss = data["cov_loss"] 
    y_inv_cov_loss = data["inv_cov_loss"]

    xMax = max(x)
    xMin = min(x)
    yMax = max( max(y_cov_loss), max(y_inv_cov_loss))
    tol = 10
    xStart = xMin-tol

    plt.plot(x, y_cov_loss, label="Covariance Matrix Loss")
    plt.plot(x, y_inv_cov_loss, label="Inverse Covariance Matrix Loss")
    plt.legend(framealpha=1, frameon=True)

    plt.xlabel('Window size')
    plt.ylabel('Losses')

    plt.axis([xStart, xMax, 0, yMax+tol])
    plt.axhline(0, color='grey')
    plt.axvline(xStart+1, color='grey')

    plt.title(title)
    plt.savefig(imagefile)
    plt.close()

# conda install -c conda-forge ggplot

if __name__ == '__main__':
    seed = 100
    np.random.seed(seed)
    X = np.random.rand(10000000,15)
    print ("+++++experiment1++++++++")
    data = experiment1(X)
    title = 'Static window size vs Update Window Size Threshold 1'
    imagefile = "chart1.png"
    draw(data, title, imagefile)
    print ("+++++experiment2+++++++")
    data = experiment2(X)
    title = 'Static window size vs Update Window Size Threshold 2'
    imagefile = "chart2.png"
    draw(data, title, imagefile)

