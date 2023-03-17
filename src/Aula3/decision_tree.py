import numpy as np
import collections

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index # índice do atributo que será testado
        self.threshold = threshold # valor de limite para o teste do atributo
        self.left = left # sub-árvore à esquerda
        self.right = right # sub-árvore à direita
        self.value = value # valor da classe, caso seja um nó folha

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', prune=None, size=None, independence=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.prune = prune
        self.size = size   # pre-prunning
        self.independence = independence   # pre-prunning
        self.tree = None
        
    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y, depth=0)
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _predict(self, inputs):
        node = self.tree
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def score(self, X, y):
        """
        Retorna a precisão do modelo no conjunto de dados fornecido.
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy

    
    def _get_leaf_nodes(self, node):
        """
        Retorna todos os nós folha da subárvore enraizada em node.
        """
        if node is None:
            return []
        if node.value is not None:
            return [node]
        leaf_nodes_left = self._get_leaf_nodes(node.left)
        leaf_nodes_right = self._get_leaf_nodes(node.right)
        return leaf_nodes_left + leaf_nodes_right
    
    def _get_nodes_list(self):
        """
        Retorna uma lista com todos os nós da árvore.
        """
        nodes_list = []
        queue = collections.deque()
        queue.append(self.tree)
        while len(queue) > 0:
            node = queue.popleft()
            if node is not None:
                nodes_list.append(node)
                queue.append(node.left)
                queue.append(node.right)
        return nodes_list


    def reduced_error_pruning(self, X_val, y_val):
        if self.tree is None:
            raise ValueError("You need to fit the model before pruning.")
        nodes = self._get_nodes_list()
        best_accuracy = self.score(X_val, y_val)
        for node in reversed(nodes):
            node_left = node.left
            node_right = node.right
            node.left = None
            node.right = None
            accuracy = self.score(X_val, y_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                continue
            node.left = node_left
            node.right = node_right
        return self


    """
    Pre-prunning: max-depth, independence, size
    Post-prunning: 
    Tanto o Reduced Error Pruning (REP) quanto o Pessimistic Error Pruning (PEP) são métodos post-prunning, o que significa que são aplicados após treinar a árvore de decisão.
        1. Reduced Error Pruning (REP): criar uma lista de todos os nós da árvore e, em seguida, percorrer a lista de trás para frente. Em cada iteração, removeremos o nó da lista e avaliaremos a precisão da árvore no conjunto de validação. Se a precisão for melhor, o nó é removido da árvore.
           Caso contrário, o nó é adicionado de volta à lista e a iteração continua para o próximo nó na lista.
        
        2. Pessimistic Error Pruning (PEP): adicionar uma nova variável de classe pep_limit que define a margem de erro máxima permitida.
           Em seguida, vamos adicionar uma condição no método _grow_tree para calcular a margem de erro do nó interno antes de criar seus filhos.
           Se a margem de erro do nó interno for maior do que o limite pep_limit, o nó será transformado em um nó folha com a classe mais frequente.
    """
    
    def _grow_tree(self, X, y, depth=0, parent=None):

        if depth == self.max_depth:   # adiciona a condição de profundidade máxima
            return Node(value=self._most_common_class(y))
        
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split or n_samples < self.min_samples_leaf:   # adiciona a condição de tamanho mínimo do nó
            return Node(value=self._most_common_class(y))


        # PRE-PRUNNING
        if self.independence is not None and parent is not None:  # adiciona a condição de independência
            if self._check_independence(X, parent):
                return Node(value=self._most_common_class(y))
                
        if self.size is not None and depth > 0:  # adiciona a condição de tamanho máximo da árvore
            if self._get_tree_size(X, y) >= self.size:
                return Node(value=self._most_common_class(y))

        best_feature_index, best_threshold = self._choose_best_feature(X, y)

        left_idx = X[:, best_feature_index] <= best_threshold
        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[~left_idx], y[~left_idx]

        left = self._grow_tree(X_left, y_left, depth=depth+1, parent=X)   # passa o nó pai como argumento
        right = self._grow_tree(X_right, y_right, depth=depth+1, parent=X)   # passa o nó pai como argumento

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left, right=right)
    

   
    """
    Para implementar o pre-prunning de tamanho (size), precisamos verificar o tamanho da árvore antes de fazer uma divisão.
    Se o tamanho atual da árvore for maior ou igual ao tamanho máximo definido pelo usuário, retornamos um nó folha com a classe mais frequente.
    """
    def _get_tree_size(self, node):
        if node is None:
            return 0
        return 1 + self._get_tree_size(node.left) + self._get_tree_size(node.right)
    
    def _gini_impurity(self, left_probabilities, right_probabilities, n_left, n_right):
        # compute gini impurity
        left_gini = 1 - sum(np.square(left_probabilities))
        right_gini = 1 - sum(np.square(right_probabilities))
        weighted_gini = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)
        return weighted_gini
    

    def _choose_best_feature(self, X, y):
        if self.criterion == 'entropy':
            best_feature_index, best_threshold = self._choose_best_feature_entropy(X, y)
        elif self.criterion == 'gini':
            best_feature_index, best_threshold = self._choose_best_feature_gini(X, y)
        else: # GAIN RATIO
            best_feature_index, best_threshold = self._choose_best_feature_gain_ratio(X, y)
        return best_feature_index, best_threshold