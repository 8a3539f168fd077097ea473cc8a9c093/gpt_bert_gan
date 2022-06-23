from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np


def plot_ratings(rating_bins):
    """
    Show 6 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]
    rating_bins5 = rating_bins[5]
    rating_bins6 = rating_bins[6]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.3, rating_bins0, 0.1, label='epoch 0')
    plt.bar(X_axis - 0.2, rating_bins1, 0.1, label='epoch 1')
    plt.bar(X_axis - 0.1, rating_bins2, 0.1, label='epoch 2')
    plt.bar(X_axis + 0, rating_bins3, 0.1, label='epoch 3')
    plt.bar(X_axis + 0.1, rating_bins4, 0.1, label='epoch 4')
    plt.bar(X_axis + 0.2, rating_bins5, 0.1, label='epoch 5')
    plt.bar(X_axis + 0.3, rating_bins6, 0.1, label='epoch 6')


    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)

def plot_ratings_compared(rating_bins):

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]


    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.1, rating_bins0, 0.1, label='base pretrained GPT-2')
    plt.bar(X_axis + 0.1, rating_bins1, 0.1, label='fine-tuned nice GPT-2')


    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("sentiment compared with emnlp_news dataset context")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)

def plot_ratings_12(rating_bins):
    """
    Show 12 iterations of training with bar plots.
    """

    X = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]
    rating_bins5 = rating_bins[5]
    rating_bins6 = rating_bins[6]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.3, rating_bins0, 0.1, label='epoch 0')
    plt.bar(X_axis - 0.2, rating_bins1, 0.1, label='epoch 2')
    plt.bar(X_axis - 0.1, rating_bins2, 0.1, label='epoch 4')
    plt.bar(X_axis + 0, rating_bins3, 0.1, label='epoch 6')
    plt.bar(X_axis + 0.1, rating_bins4, 0.1, label='epoch 8')
    plt.bar(X_axis + 0.2, rating_bins5, 0.1, label='epoch 10')
    plt.bar(X_axis + 0.3, rating_bins6, 0.1, label='epoch 12')

    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("ratings compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/ratings{}.png'.format(log_time_str)
    plt.savefig(file)



def plot_fake_training(rating_bins):
    """
    Show gen fake training
    """

    X = ['FAKE', 'TRUE']
    rating_bins0 = rating_bins[0]
    rating_bins1 = rating_bins[1]
    rating_bins2 = rating_bins[2]
    rating_bins3 = rating_bins[3]
    rating_bins4 = rating_bins[4]
    rating_bins5 = rating_bins[5]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.3, rating_bins0, 0.1, label='epoch 0')
    plt.bar(X_axis - 0.2, rating_bins1, 0.1, label='epoch 1')
    plt.bar(X_axis - 0.1, rating_bins2, 0.1, label='epoch 2')
    plt.bar(X_axis + 0, rating_bins3, 0.1, label='epoch 3')
    plt.bar(X_axis + 0.1, rating_bins4, 0.1, label='epoch 4')
    plt.bar(X_axis + 0.2, rating_bins5, 0.1, label='epoch 5')


    plt.xticks(X_axis, X)
    plt.xlabel("rating ")
    plt.ylabel("frequency")
    plt.title("fake/true compared at each epoch")
    plt.legend()
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'visual/saved_plots/bert_fake/ratings{}.png'.format(log_time_str)
    plt.savefig(file)

def plot_negativity_evolution(arr1, arr2, arr3, arr4, arr5):

    training_epochs = [i for i in range(14)]

    # Assign variables to the y axis part of the curve
    negativity_1 = arr1
    negativity_2 = arr2
    negativity_3 = arr3
    negativity_4 = arr4
    negativity_5 = arr5

    # Plotting both the curves simultaneously
    plt.plot(training_epochs, negativity_1, color='r', label="run 1")
    plt.plot(training_epochs, negativity_2, color='g', label="run 2")
    plt.plot(training_epochs, negativity_3, color='b', label="run 3")
    plt.plot(training_epochs, negativity_4, color='y', label="run 4")
    plt.plot(training_epochs, negativity_5, color='c', label="run 5")


    plt.ylim(ymin=0, ymax=5000)
    plt.xlim(xmin=0)
    plt.xticks([i for i in range(14) if i % 2 == 0])

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Epochs")
    plt.ylabel("Negativity")
    plt.title("GPT-2 training with rewards adamW and optimizations")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # save the file with a data
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'saved_plots/neg_evo_plot{}.png'.format(log_time_str)
    plt.savefig(file)

def plot_BLEU_evolution(arr1, arr2, arr3):

    # Using Numpy to create an array X
    training_epochs = [i for i in range(33)]

    # Assign variables to the y axis part of the curve
    negativity_1 = arr1
    negativity_2 = arr2
    negativity_3 = arr3

    # Plotting both the curves simultaneously
    plt.plot(training_epochs, negativity_1, color='r', label="image coco, 12 layers")
    plt.plot(training_epochs, negativity_2, color='g', label="image coco, 3 layers")
    plt.plot(training_epochs, negativity_3, color='b', label="EMNLP news, 6 layers")

    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0)
    plt.xticks([i for i in range(33) if i % 2 == 0])

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Epochs")
    plt.ylabel("2-gram BLEU score")
    plt.title("evolution of 2-gram BLEU score with 3 different setups")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend(loc='lower right')

    # save the file with a data
    log_time_str = strftime("%m_%d_%H%M", localtime())
    file = 'saved_plots/BLEU_evo_plot{}.png'.format(log_time_str)
    plt.savefig(file)



"""
fake_true_bins = [[8808, 1192], [8714, 1268], [7435, 2545],
                  [7638, 2362], [8014, 1986], [8296, 1704]]
plot_fake_training(fake_true_bins)
"""
"""
neg1 = np.array([1395, 1384, 985, 652, 481, 460, 333, 299, 303, 369, 333, 315, 389, 366, 327, 253, None, None, None,None,None])
neg2 = np.array([1389, 1815, 2028, 2981, 3378, 2176, 1070, 569, 1030, 1062, 1398, 1123, None ,None ,None ,None ,None ,None, None,None,None])
neg3 = np.array([1375, 1368, 1327, 1349, 1346, 1405, 1320, 1317, 1358, 1426, 1373, 1374, 1319, 1327, 1335, 1360, 1341, 1387, 1370, 1316,None])
plot_negativity_evolution(neg1, neg2, neg3)
"""
"""
BLEU1 = [0.721, 0.688, 0.231, 0.057, 0.047, 0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047
         ,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047,0.047]
BLEU2 = [0.411, 0.404, 0.42, 0.435, 0.398, 0.402, 0.368, 0.415, 0.428, 0.406, 0.399, 0.445, 0.419, 0.381, 0.429, 0.407,
         0.436, 0.435, 0.415, 0.401, 0.416, 0.386, 0.405, 0.412, 0.414, 0.395, 0.42, 0.401, None, None, None, None, None]
BLEU3 = [0.92, 0.928, 0.927, 0.925, 0.915, 0.926, 0.929, 0.927, 0.922, 0.918, 0.925, 0.927, 0.922, 0.917, 0.921, 0.919,
         0.921, 0.922, 0.932, 0.918, 0.923, 0.919, 0.915, 0.915, 0.914, 0.919, 0.923, 0.916,0.919, 0.915, 0.913, 0.918,
         0.912]
plot_BLEU_evolution(BLEU1, BLEU2, BLEU3)
"""
"""
rating_bins = [[20253, 61553, 18194], [14374, 69439, 16187]]
plot_ratings_compared(rating_bins)
"""

""""
neg1 = [1391, 1379, 1408, 1322, 1208, 1299, 1323, 1426, 1381, 1367, 1441, 1427, 1423, 1384]
neg2 = [1378, 1337, 1283, 1218, 1095, 954, 968, 990, 1049, 1092, 1079, 1086, 1107, 1114]
neg3 = [1394, 1318, 1084, 1020, 771, 334, 368, 452, 427, 393, 431, 432, 411, 420]
neg4 = [1388, 1341, 1090, 747, 418, 481, 485, 508, 514, 419, 400, 394, 374, 348]
neg5 = [1375, 1476, 2145, 2255, 2342, 2613, 2548, 2551, 2809, 2793, 2149, 1156, 614, 462]
plot_negativity_evolution(neg1, neg2, neg3, neg4, neg5)
"""

"""
neg1 = [1395, 1384, 985, 652, 481, 460, 333, 299, 303, 369, 333, 315, 389, 366]
neg2 = [1389, 1815, 2028, 2981, 3378, 2176, 1070, 569, 1030, 1062, 1398, 1123, None, None]
neg3 = [1375, 1368, 1327, 1349, 1346, 1405, 1320, 1317, 1358, 1426, 1373, 1374, 1319, None]
neg4 = [1391, 1344, 1082, 1016, 977, 984, 1026, 958, 958, 768, 642, 602, 664, 644]
neg5 = [1384, 1304, 1173, 859, 943, 951, 827, 733, 706, 618, 540, 458, 516, 523]
plot_negativity_evolution(neg1, neg2, neg3, neg4, neg5)
"""