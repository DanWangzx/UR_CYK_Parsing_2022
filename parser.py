import enum
import torch


class CYK(object):
    def __init__(self, weights_file=None) -> None:
        self.nts = {'S':3, 'A':0, 'B':1, 'C':2} # nonterminal nodes
        self.d_nts = {3:'S', 0:'A', 1:'B', 2:'C'}
        self.ts = {'a':0, 'b':1, 'c':2}
        self.w = torch.zeros(4, 4, 4, 4) # [type of rules, first symbol, second symbol, third symbol(optional)]

        if weights_file:
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
        s = True

    def decode(self, input):
        deltas = torch.zeros(len(input)+1, len(input)+1, len(self.nts))
        backtracking = {} #a_b_c_i_j_K
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
        
        # recursion function:
        def reverse(cur_key) -> str:
            if cur_key not in backtracking:

                return '(' + cur_key.split('_')[0] + ' ' + str((input[int(cur_key.split('_')[1])]))+ ')'
            key1, key2 = backtracking[cur_key]
            out = ''
            sub_out_1 = reverse(key1)
            sub_out_2 = reverse(key2)
            out = '(' + cur_key.split('_')[0] + ' ' + sub_out_1 + ' ' + sub_out_2 + ')'
            return out 

        out_string = reverse(last_key)
        return out_score, out_string


# test modules

import sys
filename = sys.argv[1]
test_cases = sys.argv[2]
cyk = CYK(filename)
#cyk.decode(input_string)
#cyk.decode(input_string_02)
with open(test_cases, 'r') as f:
    for i in f:
        i = i.strip()
        input_string = []
        for letter in i:
            input_string.append(letter)
        print(input_string)
        print(cyk.decode(input_string))

