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

    def _choose_best_feature_entropy(self, X, y):
        best_feature_index = None
        best_threshold = None
        best_info_gain = -1

        n_samples, n_features = X.shape
        H_parent = self._entropy(y)

        for feature_index in range(n_features):
            # ordena as amostras com base no valor do atributo
            X_feature = X[:, feature_index]
            sorted_idx = X_feature.argsort()
            X_sorted = X[sorted_idx]
            y_sorted = y[sorted_idx]

            # encontra os possíveis pontos de divisão
            split_idx = np.where(y_sorted[:-1] != y_sorted[1:])[0] + 1
            if len(split_idx) == 0:
                continue

            # calcula a entropia para cada ponto de divisão
            for threshold in X_sorted[split_idx]:
                y_left = y_sorted[X_sorted <= threshold]
                y_right = y_sorted[X_sorted > threshold]
                n_left = len(y_left)
                n_right = len(y_right)
                H_children = (n_left / n_samples) * self._entropy(y_left) + (n_right / n_samples) * self._entropy(y_right)
                info_gain = H_parent - H_children

                # atualiza o melhor atributo e ponto de divisão
                if info_gain > best_info_gain:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_info_gain = info_gain

        return best_feature_index, best_threshold


    def _choose_best_feature_gini(self, X, y):
        best_score = 1.0
        best_feature_index = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                y_left = y[left_indices]
                y_right = y[~left_indices]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_probabilities = np.array([len(y_left[y_left == c]) / len(y_left) for c in range(self.n_classes)])
                right_probabilities = np.array([len(y_right[y_right == c]) / len(y_right) for c in range(self.n_classes)])

                score = self._gini_impurity(left_probabilities, right_probabilities, len(y_left), len(y_right))

                if score < best_score:
                    best_score = score
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

        


    def _choose_best_feature_gain_ratio(self, X, y):
        # escolhe o melhor atributo com base na razão de ganho
        best_feature = None
        best_gain_ratio = -1
        
        # calcula o ganho de informação de cada atributo
        for feature_index in range(self.n_features):
            gain_ratio = self._information_gain_ratio(X, y, feature_index)
            if gain_ratio > best_gain_ratio:
                best_feature = feature_index
                best_gain_ratio = gain_ratio
        
        return best_feature


    def _most_common_class(self, y):
        # calcula a classe mais frequente
        counter = collections.Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common



def main():
    # exemplo de conjunto de dados
    X_train = np.array([[1, 2], [2, 1], [2, 3], [3, 2], [4, 2], [3, 3], [2, 2], [3, 1]])
    y_train = np.array([0, 0, 1, 1, 1, 0, 0, 1])

    X_val = np.array([[1, 3], [4, 1], [2, 3], [3, 3]])
    y_val = np.array([1, 1, 0, 1])


    X_test = np.array([[1, 1], [4, 3], [3, 4], [2, 1]])

    # cria o modelo de árvore de decisão com os hiperparâmetros escolhidos
    dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion='gini', prune=None, size=None, independence=None)

    # treina o modelo com o conjunto de dados de treinamento
    dtc.fit(X_train, y_train)

    # previsão das classes dos dados de teste
    y_pred = dtc.predict(X_test)

    # fazer reduced error pruning na decision tree
    #dtc.reduced_error_pruning(X_val, y_val)

    # fazer pessimistic error pruning na decision tree
    #dtc.pessimistic_error_pruning(X_val, y_val)


    # compara as classes reais com as classes previstas
    #print("Classes reais: ", [1, 1, 0, 0])
    print("Classes previstas: ", y_pred)


main()