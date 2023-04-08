class Apriori:
    def __init__(self, minsupport):
        self.minsupport = minsupport
        self.C1 = {}
        self.L = []
        self.k = 2
    
    def fit(self, filepath):
        """Find frequent itemsets in the given file"""
        self._read_file(filepath)
        self.L1 = self._apriori_prune(self.C1)
        self.L = self.L1.copy()
        print('====================================')
        print('Frequent 1-itemset is', self.L1)
        print('====================================')
        while self.L:
            C = self._apriori_count_subset(self.L)
            fruquent_itemset = self._apriori_prune(C)
            if not fruquent_itemset:
                break
            print('====================================')
            print(f'Frequent {self.k}-itemset is', fruquent_itemset)
            print('====================================')
            self.L = self._apriori_gen(fruquent_itemset)
            self.k += 1

    def _read_file(self, filepath):
        """Read the transaction data from a file"""
        with open(filepath) as file:
            for line in file:
                for item in line.split():
                    if item in self.C1:
                        self.C1[item] += 1
                    else:
                        self.C1[item] = 1
        self.L1 = self._apriori_prune(self.C1)

    def _apriori_gen(self, itemset):
        """Generate new candidate itemsets by joining existing ones"""
        candidate = []
        for i in range(len(itemset)):
            for j in range(i+1, len(itemset)):
                element = itemset[i]
                element1 = itemset[j]
                if element[0:(len(element)-1)] == element1[0:(len(element1)-1)]:
                    unionset = element[0:(len(element)-1)] + element1[len(element1)-1] + element[len(element)-1]  # Combine (k-1)-Itemset to k-Itemset 
                    unionset = ''.join(sorted(unionset))  # Sort itemset by dict order
                    candidate.append(unionset)
        return candidate