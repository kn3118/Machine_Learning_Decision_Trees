# Import numpy library
import numpy as np
#import pydot
import copy
import time

""" HELPER FUNCTIONS """

def is_leaf(dataset):
    if len(np.unique(dataset[:,-1])) == 1:
        return 1
    else:
        return 0

def split_dataset(dataset, split):
    """
    Split dataset based on the split point
    """
    ordered = dataset[dataset[:,split[0]].argsort()]
    l_dataset = ordered[ordered[:,split[0]] <= split[1]]
    r_dataset = ordered[~(ordered[:,split[0]] <= split[1])]
    return l_dataset, r_dataset

def randomise_split_data(dataset, k):
    """
    Split dataset into k parts. For a 10 fold, k = 10
    """
    random_data = np.copy(dataset)
    np.random.shuffle(random_data)
    data_split = np.split(random_data, k)
    return data_split

def train_test_data(data_split, index):
    """
    Create training and test dataset
    """
    training = np.vstack(data_split[n] for n in [x for x in range(len(data_split)) if x != index])
    test = data_split[index]
    return training, test

def print_confusion_matrix(cm, cols=["1", "2", "3", "4"], rows=["1", "2", "3", "4"]):
    if (len(cols) != cm.shape[1]) or (len(rows) != cm.shape[0]):
        print("Shapes do not match")
        return
    s = cm.__repr__()
    s = s.split("array(")[1]
    s = s.replace("      ", "")
    s = s.replace("[[", " [")
    s = s.replace("]])", "]")
    pos = [i for i, ltr in enumerate(s.splitlines()[0]) if ltr == ","]
    pos[-1] = pos[-1] - 1
    empty = " " * len(s.splitlines()[0])
    s = s.replace("],", "]")
    s = s.replace(",", "")
    lines = []
    for i, l in enumerate(s.splitlines()):
        lines.append(rows[i] + l)
    s = "\n".join(lines)
    empty = list(empty)
    for i, p in enumerate(pos):
        empty[p - i] = cols[i]
    s = "".join(empty) + "\n" + s
    print(s)

def average_metrics(data_list):
    result = []

    data = np.array(data_list)
    for i in range(data.shape[1]):
        result.append(data[:,i].mean())

    return result

def confusion_matrix(test_db_predicted, test_db_label):
    confusionMatrix = np.zeros((4,4))
    for i in range(len(test_db_label)):
        row = int(test_db_label[i] - 1)
        col = int(test_db_predicted[i] - 1)
        confusionMatrix[row][col] = confusionMatrix[row][col] + 1

    return confusionMatrix

#def walk_dictionary(graph, dictionary, parent_node=None):
#    if parent_node is not None:
#        if dictionary['leaf'] == 0:
#            from_name = parent_node.get_name().replace("\"", "") + '_' + "X" + str(dictionary['attribute']) + " < " + str(dictionary['value'])
#            from_label = "X" + str(dictionary['attribute']) + " < " + str(dictionary['value'])
#            node_from = pydot.Node(from_name, label=from_label)
#            graph.add_node(node_from)
#            graph.add_edge( pydot.Edge(parent_node, node_from) )
#
#            if isinstance(dictionary['left'], dict):
#                walk_dictionary(graph, dictionary['left'], node_from)
#            if isinstance(dictionary['right'], dict):
#                walk_dictionary(graph, dictionary['right'], node_from)
#        else:
#            from_name = parent_node.get_name()
#            from_label = parent_node.get_name()
#            node_from = pydot.Node(from_name, label=from_label)
#            to_name = parent_node.get_name().replace("\"", "") + '_' + str(dictionary['value'])
#            to_label = str(dictionary['value'])
#            node_to = pydot.Node(to_name, label=to_label, shape='box')
#            graph.add_node(node_to)
#            graph.add_edge(pydot.Edge(node_from, node_to))
#
#    else:
#        from_name = "X" + str(dictionary['attribute']) + " < " + str(dictionary['value'])
#        from_label = "X" + str(dictionary['attribute']) + " < " + str(dictionary['value'])
#        node_from = pydot.Node(from_name, label=from_label)
#        graph.add_node(node_from)
#        walk_dictionary(graph, dictionary['left'], node_from)
#        walk_dictionary(graph, dictionary['right'], node_from)

#def plot_tree(tree, name):
#    graph = pydot.Dot(graph_type='graph')
#    walk_dictionary(graph, tree)
#    graph.write_png(name + '.png')

def entropy(dataset):
    """
    Measure level of uncertainty of a dataset
    """
    outputs, freq = np.unique(dataset[:,-1], return_counts = True)
    entropy = -np.sum([(freq[i]/np.sum(freq)) * np.log2(freq[i]/np.sum(freq)) for i in range(outputs.shape[0])])
    return entropy

def info_gain(dataset, l_dataset, r_dataset):
    """
    Measure the reduction of information entropy
    """
    entropy_all = entropy(dataset)
    entropy_left = entropy(l_dataset)
    entropy_right = entropy(r_dataset)
    t_rows = dataset.shape[0]
    l_rows = l_dataset.shape[0]
    r_rows = r_dataset.shape[0]
    remainder = ((l_rows / t_rows) * entropy_left + (r_rows / t_rows) * entropy_right)
    gain = entropy_all - remainder
    return gain



""" MAIN FUNCTIONS """

def find_split(dataset):
    """
    Find the split point
    """
    max_gain = 0
    best_split = [0, 0]

    for attr in range(dataset.shape[1] - 1):
    # Sort dataset by this attribute column
        ordered = dataset[dataset[:,attr].argsort()]

        for row in range(1, ordered.shape[0]):
            if ordered[row, attr] != ordered[row - 1, attr]:          
                value = ordered[row - 1, attr]                        
                split = [attr, value]
                l_dataset, r_dataset = split_dataset(dataset, split)

                gain = info_gain(ordered, l_dataset, r_dataset)

                if gain > max_gain:
                    max_gain = gain
                    best_split = split

    if best_split[1] == 0:
        print(dataset.shape[0])
        print(dataset)

    return best_split

def decision_tree_learning(dataset, depth):
    """
    Build a decision tree based on a given dataset
    """
    if is_leaf(dataset):
        node = {'value': dataset[0,-1], 'leaf': 1}
        return node, depth
    else:
        split = find_split(dataset)
        if split[1] == 0:
            elements, count = np.unique(dataset[:,-1], return_counts = True)
            max_count_index = np.where(count == max(count))[0][0]
            max_element = int(elements[max_count_index])
            node = {'value': max_element , 'leaf': 1}
            return node, depth
        l_dataset, r_dataset = split_dataset(dataset,split)
        node = {'attribute': split[0], 'value': split[1], 'left': {}, 'right': {}, 'leaf': 0}
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        node['left'] = l_branch
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        node['right'] = r_branch
        return node, max([l_depth, r_depth])

def cross_validation(dataset, k=10):
    """
    Given a dataset, we perform cross_validation (by default 10 fold) both
    before pruning and after pruning. 
    
    1) Split dataset into k-fold parts
    2) Create train_val and test set
    3) Build an unpruned tree using train_val dataset and evaluate the unpruned trained tree
    4) Take the train_val dataset and split it into train and val dataset
    5) Build an unpruned tree using the train dataset
    6) Prune the tree using val dataset and evaluate the pruned trained tree
    
    Output: 
    Return confusion matrix and average of all the metrics 
    (Accuracy, Precision, Recall, F1-score and Depth) for before and after pruning
    """
    splits = randomise_split_data(dataset, k)

    u_accuracy = []
    u_precision = []
    u_recall = []
    u_F1 = []
    u_c_matrix = []
    u_depth = []

    p_accuracy = []
    p_precision = []
    p_recall = []
    p_F1 = []
    p_c_matrix = []
    p_depth = []

    t_fold = []
    for i in range(k):
        t0 = time.time()
        train_val, test = train_test_data(splits, i)

        # Train tree on train_val and get performance metrics
        unpruned_tree, depth = decision_tree_learning(train_val, 0)
        accuracy, precision, recall, F1, c_matrix = evaluate(test, unpruned_tree)

        u_accuracy.append(accuracy)
        u_precision.append(precision)
        u_recall.append(recall)
        u_F1.append(F1)
        u_c_matrix.append(c_matrix)
        u_depth.append(depth)
        
        
        for j in range(k-1):
            if i == 0:
                print("\rTest fold = {:2d}/{}\t|\tPrune/Val fold = {}/{}".format(i + 1, k, j + 1, k - 1), end="")
            else:
                print("\rTest fold = {:2d}/{}\t|\tPrune/Val fold = {}/{}\t|\tTime of last fold = {:0.2f}s\t|\tAvg time per fold = {:0.2f}s".format(i + 1, k, j + 1, k - 1, t_fold[-1], np.mean(t_fold)), end="")
            train = np.vstack(splits[n] for n in [x for x in range(k) if x not in [i, j]])
            val = splits[j]

            # Train a tree on the train set
            unpruned, _ = decision_tree_learning(train, 0)

            # Prune previous tree until convergence (no more pruning happens)
            changes = []
            tree_to_prune = copy.deepcopy(unpruned)
            while changes != 0:
                pruned_tree, changes = prune_tree(tree_to_prune, val)
                tree_to_prune = copy.deepcopy(pruned_tree)

            accuracy, precision, recall, F1, c_matrix = evaluate(test, pruned_tree)
            depth = calculate_depth(pruned_tree)

            p_accuracy.append(accuracy)
            p_precision.append(precision)
            p_recall.append(recall)
            p_F1.append(F1)
            p_c_matrix.append(c_matrix)
            p_depth.append(depth)

        t_fold.append(time.time() - t0)

    avg_acc_bp = np.mean(u_accuracy)
    avg_acc_ap = np.mean(p_accuracy)

    avg_pr_per_class_bp = np.mean(u_precision, axis=0)[0]
    avg_pr_per_class_ap = np.mean(p_precision, axis=0)[0]

    avg_re_per_class_bp = np.mean(u_recall, axis=0)[0]
    avg_re_per_class_ap = np.mean(p_recall, axis=0)[0]

    avg_F1_per_class_bp = np.mean(u_F1, axis=0)[0]
    avg_F1_per_class_ap = np.mean(p_F1, axis=0)[0]

    avg_confusion_matrix_bp = np.array(u_c_matrix).mean(axis=0)
    avg_confusion_matrix_ap = np.array(p_c_matrix).mean(axis=0)
    
    std_acc_bp = np.std(u_accuracy)
    std_acc_ap = np.std(p_accuracy)
    
    avg_depth_bp = np.mean(u_depth)
    avg_depth_ap = np.mean(p_depth)

    return avg_acc_bp, avg_acc_ap, std_acc_bp, std_acc_ap, avg_pr_per_class_bp, avg_pr_per_class_ap, avg_re_per_class_bp, avg_re_per_class_ap, avg_F1_per_class_bp,\
           avg_F1_per_class_ap, avg_confusion_matrix_bp, avg_confusion_matrix_ap,avg_depth_bp, avg_depth_ap

def calculate_depth(pruned_tree):
    if pruned_tree['leaf'] == 1:
        return 0
    else:
        l_depth = calculate_depth(pruned_tree['left'])
        r_depth = calculate_depth(pruned_tree['right'])
        
        if(l_depth > r_depth):
            return l_depth + 1
        else:
            return r_depth + 1

def evaluate_pruning(test_db, tree):
    test_db_predicted = []
    X_test = test_db[:, :-1]
    y_test = test_db[:, -1]

    for i in range(len(X_test)):
        y = predict(tree, X_test[i])
        test_db_predicted.append(y)

    correct = np.count_nonzero(test_db_predicted == y_test)
    return correct

def evaluate(test_db, trained_tree, mode='usual'):
    """
    Given a test set and trained decision tree, we evaluate the accuracy, precision,
    recall and F1-score of the predictions
    """
    test_db_predicted = []
    test_db_attribute = test_db[:,:-1]
    test_db_label = test_db[:,-1]

    for i in range(len(test_db_attribute)):
        y = predict(trained_tree, test_db_attribute[i])
        test_db_predicted.append(y)

    # If on pruning mode, only care about number of correct.
    if mode == 'pruning':
        correct = np.count_nonzero(test_db_predicted == test_db_label)
        return correct

    cm = confusion_matrix(test_db_predicted, test_db_label)

    recall = []
    precision = []
    F1 = []
    F1_per_fold = []
    recall_per_class = []
    precision_per_class = []
    F1_per_class = []

    for i in range(len(cm)):
        TP = cm[i][i]
        TN = -TP
        FN = -TP
        FP = -TP
        for j in range(len(cm)):
            TN = TN + cm[j,j]
            FN = FN + cm[i,j]
            FP = FP + cm[j,i]
        re = TP / (TP + FN)
        pr = TP / (TP + FP)
        f = (2 * pr * re)/(pr + re)
        recall_per_class.append(re)
        precision_per_class.append(pr)
        F1_per_class.append(f)
    recall.append(recall_per_class)
    precision.append(precision_per_class)
    F1.append(F1_per_class)
    F1_per_fold.append(np.mean(F1_per_class))

    diagonal = np.trace(cm)
    total = np.sum(cm)
    accuracy = diagonal / total

    return accuracy, precision, recall, F1, cm

def predict(node, row):
    """
    Given a set of features (row) our trained tree (node) can predict the class
    rooms
    """
    # If node is not a leaf
    if node['leaf'] != 1:
        if row[node['attribute']] < node['value']:
            if node['leaf'] == 0:
                return predict(node['left'], row)
            else:
                return node['value']
        else:
            if node['leaf'] == 0:
                return predict(node['right'], row)
            else:
                return node['value']
    # if node is a leaf node
    else:
        return node['value']

def prune_tree(node, relevant_val_dataset, changes=0):
    if node['left']['leaf'] == 1 and node['right']['leaf'] == 1:

        current_correct = evaluate_pruning(relevant_val_dataset, node)
        left_correct = evaluate_pruning(relevant_val_dataset, node['left'])
        right_correct = evaluate_pruning(relevant_val_dataset, node['right'])

        if left_correct >= current_correct and left_correct >= right_correct:
            node = node['left']
            changes += 1
        elif right_correct > current_correct and right_correct > left_correct:
            node = node['right']
            changes += 1
    else:
        node_split = [node['attribute'], node['value']]
        l_dataset, r_dataset = split_dataset(relevant_val_dataset, node_split)
        if node['left']['leaf'] == 0:
            node['left'], changes = prune_tree(node['left'], l_dataset, changes)
        if node['right']['leaf'] == 0:
            node['right'], changes = prune_tree(node['right'], r_dataset, changes)

    return node, changes

if __name__ == "__main__":
    np.random.seed(2)
    
    # LOAD DATASET
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    
    # CROSS VALIDATION
    k = 10
    avg_acc_bp, avg_acc_ap, std_acc_bp, std_acc_ap, \
    avg_pr_per_class_bp, avg_pr_per_class_ap, \
    avg_re_per_class_bp, avg_re_per_class_ap, \
    avg_F1_per_class_bp, avg_F1_per_class_ap, \
    avg_confusion_matrix_bp, avg_confusion_matrix_ap, \
    avg_depth_bp, avg_depth_ap = cross_validation(dataset, k)

    # OUTPUT RESULTS ON TERMINAL
    print("\n\nCROSS-VALIDATION PERFORMANCE")
    print("\n")
    print('Metrics before pruning')
    print("________________________________________________________")
    print('AVERAGE ACCURACY:                {:0.4f}'.format(avg_acc_bp))
    print('STD ACCURACY:                    {:0.4f}'.format(std_acc_bp)) 
    print('AVERAGE PRECISION (per class):   {}'.format(np.around(avg_pr_per_class_bp,3)))
    print('AVERAGE RECALL    (per class):   {}'.format(np.around(avg_re_per_class_bp,3)))
    print('AVERAGE F1 SCORE  (per class):   {}'.format(np.around(avg_F1_per_class_bp,3)))
    print('AVERAGE CONFUSION MATRIX')
    print_confusion_matrix(np.around(avg_confusion_matrix_bp,1))
    print('AVERAGE MAXIMUM DEPTH:           {:0.1f}'.format(avg_depth_bp))
    print("________________________________________________________\n\n")
    print('Metrics after pruning (per class)')
    print("________________________________________________________")
    print('AVERAGE ACCURACY:                {:0.4f}'.format(avg_acc_ap))
    print('STD ACCURACY:                    {:0.4f}'.format(std_acc_ap))
    print('AVERAGE PRECISION (per class):   {}'.format(np.around(avg_pr_per_class_ap,3)))
    print('AVERAGE RECALL    (per class):   {}'.format(np.around(avg_re_per_class_ap,3)))
    print('AVERAGE F1 SCORE  (per class):   {}'.format(np.around(avg_F1_per_class_ap,3)))
    print('AVERAGE CONFUSION MATRIX')
    print_confusion_matrix(np.around(avg_confusion_matrix_ap,1))
    print('AVERAGE MAXIMUM DEPTH:           {:0.1f}'.format(avg_depth_ap))
    print("________________________________________________________\n\n")
    print("________________________________________________________")
    print(" - ORIGINAL ACCURACY (10fold): {:0.2f}%".format(avg_acc_bp * 100))
    print(" - PRUNED ACCURACY   (10fold): {:0.2f}%".format(avg_acc_ap * 100))
