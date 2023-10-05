
import torch

class CYK(object):
    def __init__(self, weights_file=None) -> None:
        # hard-coded domains. NTS - non-ternimal nodes; d_NTS - decoder of non-terminal nodes 
        # TS - terminal nodes 
        # path - the weights that have been used and updated. 
        self.nts = {'S':3, 'A':0, 'B':1, 'C':2} 
        self.d_nts = {3:'S', 0:'A', 1:'B', 2:'C'}
        self.ts = {'a':0, 'b':1, 'c':2}
        self.path = []

        #self.w - weights, in format [type, first_symbol, second_symbol, third_symbol(optional)]
        if not weights_file:
            self.w = torch.full((4, 4, 4, 4), 1e-5) # weights cannot be initialized to zero, if we still want to have a full tree in first parse

        else:
            self.w = torch.zeros(4, 4, 4, 4)
            with open(weights_file, 'r') as f:
                for i in f.readlines():
                    string, score = i.split()
                    score = int(score)
                    tokens = string.split('_')
                    if tokens[0] == 'T':
                        self.w[0][self.nts[tokens[1]]][self.ts[tokens[2]]][:] = score
                    elif tokens[0] == 'R':
                        self.w[1][self.nts[tokens[1]]][self.nts[tokens[2]]][self.nts[tokens[3]]] = score
                    elif tokens[0] == 'F':
                        self.w[2][self.nts[tokens[1]]][self.ts[tokens[2]]][:] = score
                    elif tokens[0] == 'L':
                        self.w[3][self.nts[tokens[1]]][self.ts[tokens[2]]][:] = score

                    else:
                        raise 'Error: weights file not matching'

    def decode(self, input): #input_form: ['b', 'a', 'a', 'a', 'a', 'a', 'a', ...]
        deltas = torch.zeros(len(input)+1, len(input)+1, len(self.nts))
        backtracking = {} # {a_i_k : (b_i_j, c_j_k), ...}
        for i in range(len(input)):
            for t in range(len(self.nts)):
                deltas[i][i+1][t] = self.w[0][t][self.ts[input[i]]][0]
                    
        for span in range(2, len(input)+1):
            for i in range(0, len(input)- span+1):
                k = i + span
                for j in range(i+1, k):
                    for a in range(len(self.nts)):
                        for b in range(len(self.nts)):
                            for c in range(len(self.nts)):
                                output = deltas[i][j][b] + deltas[j][k][c] + self.w[1][a][b][c] + self.w[2][a][self.ts[input[i]]][0] + self.w[3][a][self.ts[input[k-1]]][0]
                                #print(self.d_nts[a], self.d_nts[b], self.d_nts[c], i, j, k, output)
                                if output > deltas[i][k][a]:
                                    deltas[i][k][a] = output
                                    key, sub_1, sub_2 = '', '', ''
                                    key = self.d_nts[a] + '_'+ str(i)  + '_'+ str(k) #+ '_' + self.d_nts(b) + '_'+ self.d_nts(c) + '_'+ str(j)
                                    sub_1 = self.d_nts[b] + '_' + str(i) + '_'+ str(j)
                                    sub_2 = self.d_nts[c] + '_' + str(j) + '_'+ str(k)
                                    backtracking[key] = (sub_1, sub_2)

        #backtracking with dictionary and recursion 
        out_score = deltas[0][len(input)][3]

        last_key = ''
        last_key = 'S' + '_'+ str(0)  + '_'+ str(len(input))
        
        # recursion function that outputs the (sub)tree from a node:
        def reverse(cur_key) -> str:
            if cur_key not in backtracking:
                return '(' + cur_key.split('_')[0] + ' ' + str((input[int(cur_key.split('_')[1])]))+ ')'
            key1, key2 = backtracking[cur_key]
            out = ''
            sub_out_1 = reverse(key1)
            sub_out_2 = reverse(key2)
            out = '(' + cur_key.split('_')[0] + ' ' + sub_out_1 + ' ' + sub_out_2 + ')' # building a tree from its sub-trees, in a bottom-up manner.
            return out 

        out_string = reverse(last_key)
        return out_score, out_string


    def train(self, train_file, iterations):
        with open(train_file, 'r') as f:
            c = 0
            corpus = f.readlines()
            while c <= iterations: 
                out = True
                for line in corpus:
                    line = line.rstrip()
                    # extract features and input strings from the gold parse
                    string, feature_rules = self.feature_extraction(line) #:[str, str, str ...]
                    # decode a parse w.r.t current weights
                    decoded_score, decoded_tree = self.decode(string)
                    if decoded_tree == line:
                        print(f'iter{c}, the parse is correct')
                        continue
                    else:
                        print(f'iter {c}, the current parse is wrong')
                        print(f'gold tree: {line}')
                        print(f'decoded tree: {decoded_tree}')
                        out = False
                    # extract features from decoded tree. 
                    string_2, feature_observation = self.feature_extraction(decoded_tree)
                    self.w_updates(feature_rules, feature_observation)
                    self.show()
                # if all parses in an iteration are true, then the algorithm converges. 
                if out:
                    print(f'training complete within {c} iterations, all parses are correct')
                    break
                c +=1
            print(f'iter {c} complete, the weights is returned: ')
            return self.w
    
    def show(self):
        # simple function that takes in a range of features and return their weights
        string = ''
        for tokens in self.path:
            label = tokens[0]
            key = ''
            if label  == 'T':
                s = self.w[0][self.nts[tokens[1]]][self.ts[tokens[2]]][:] 
                key = label + '_' + tokens[1] + '_' + tokens[2] + ' ' + str(round(float(s[0])))
            elif label == 'R':
                s = self.w[1][self.nts[tokens[1]]][self.nts[tokens[2]]][self.nts[tokens[3]]]
                key = label + '_' + tokens[1] + '_' + tokens[2] + '_' + tokens[3] + ' ' + str(round(float(s)))
            elif label == 'F':
                s = self.w[2][self.nts[tokens[1]]][self.ts[tokens[2]]][:]
                key = label + '_' + tokens[1] + '_' + tokens[2] + ' ' + str(round(float(s[0])))
            elif label == 'L':
                s = self.w[3][self.nts[tokens[1]]][self.ts[tokens[2]]][:]
                key = label + '_' + tokens[1] + '_' + tokens[2] + ' ' + str(round(float(s[0])))
            string += key
            string += '\n'
        print('current weight files (partial)')
        print(string)


    def w_updates(self, f_rules, f_obs): # perceptron update of weights w.r.t features
        # strengthen(add) the actual features in the gold parse
        for tokens in f_rules:
            if tokens not in self.path:
                self.path.append(tokens)
            label = tokens[0]
            if label  == 'T':
                self.w[0][self.nts[tokens[1]]][self.ts[tokens[2]]][:] += 1
            elif label == 'R':
                self.w[1][self.nts[tokens[1]]][self.nts[tokens[2]]][self.nts[tokens[3]]] += 1
            elif label == 'F':
                self.w[2][self.nts[tokens[1]]][self.ts[tokens[2]]][:] += 1
            elif label == 'L':
                self.w[3][self.nts[tokens[1]]][self.ts[tokens[2]]][:] += 1
        
        # weaken(substract) the observed features in a wrong parse
        for tokens in f_obs:
            if tokens not in self.path:
                self.path.append(tokens)
            label = tokens[0]
            if label  == 'T':
                self.w[0][self.nts[tokens[1]]][self.ts[tokens[2]]][:] -= 1
            elif label == 'R':
                self.w[1][self.nts[tokens[1]]][self.nts[tokens[2]]][self.nts[tokens[3]]] -= 1
            elif label == 'F':
                self.w[2][self.nts[tokens[1]]][self.ts[tokens[2]]][:] -= 1
            elif label == 'L':
                self.w[3][self.nts[tokens[1]]][self.ts[tokens[2]]][:] -= 1

        return

    def feature_extraction(self, tree) -> str:
        input = [*tree.replace(' ','')]
        string = []
        for i in input:
            if i not in self.nts:
                if i != '(' and i != ')':
                    string.append(i)
        rules = []
        
        # bracket mataching algorithm via stack, that returns the two most surface brackets' indices (i.e., the two immediate subtrees)
        def bracket_ins(input, left, right):
            count = 0
            bracket_list = []
            left_brs = []
            stack = 0
            for i in range(left+1, right):
                if input[i] == '(':
                    left_brs.append(i)
                    stack += 1
                elif input[i] == ')':
                    stack -= 1
                    count += 1
                    r = left_brs.pop(-1)
                    if stack == 0:
                        bracket_list.append([r, i])                    
            if count:
                return bracket_list
            return count

        # recursively update the features, in a bottom up manner
        def recursion(input, left, right):
            res = bracket_ins(input, left, right)
            cur_node = input[left+1]
            if res  == 0:
                # if there is no more subtree to diverge to 
                rules.append(('T', input[left+1], input[right-1]))
                #return the non-terminal node, and the terminal nodes. Notice, this is counted as both the left-most and right-most of the tree.
                return input[left+1], input[right-1], input[right-1] 

            else:
                br1, br2 = res
                # traverse the left and right trees, to obtain the root node of the sub-trees and the respective left-most and right-most nodes of the subtrees
                left_sub = recursion(input, br1[0], br1[1]) 
                right_sub = recursion(input, br2[0], br2[1]) 
                # S -> AB, R rules
                rules.append(('R', cur_node, left_sub[0], right_sub[0]))  
                # B -> left-most_ts, F rules
                rules.append(('F', cur_node, left_sub[1]))
                # B -> right-most_ts, L rules
                rules.append(('L', cur_node, right_sub[2]))
                return cur_node, left_sub[1], right_sub[2]

        recursion(input, 0, len(input)-1)
        print('rules learned are')
        print(rules)                
        return string, rules
        
# test modules
import sys
filename = sys.argv[1]
cyk = CYK()
cyk.train(filename, 10)
