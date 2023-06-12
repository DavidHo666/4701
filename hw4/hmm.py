from typing import Pattern, Union, Tuple, List, Dict, Any

import numpy as np
import numpy.typing as npt

"""
Some type annotations
"""
Numeric = Union[float, int, np.number, None]


"""
Global list of parts of speech
"""
POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
       'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

"""
Utility functions for reading files and sentences
"""
def read_sentence(f):
    sentence = []
    while True:
        line = f.readline()
        if not line or line == '\n':
            return sentence
        line = line.strip()
        word, tag = line.split("\t", 1)
        sentence.append((word, tag))

def read_corpus(file):
    f = open(file, 'r', encoding='utf-8')
    sentences = []
    while True:
        sentence = read_sentence(f)
        if sentence == []:
            return sentences
        sentences.append(sentence)


"""
3.1: Supervised learning
Param: data is a list of sentences, each of which is a list of (word, POS) tuples
Return: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities} 
"""
def learn_model(data:List[List[Tuple[str]]]
                ) -> Tuple[npt.NDArray, npt.NDArray, Dict[str,npt.NDArray]]:
    x0=np.array([0]*len(POS))
    for i in range(len(data)):
        _, pos = data[i][0]
        idx = POS.index(pos)
        x0[idx]+=1
    x0=x0/np.sum(x0)

    t = np.zeros([len(POS),len(POS)])
    for sentence in data:
        for i in range(len(sentence)-1):
            _, pos1 = sentence[i]
            _, pos2 = sentence[i+1]
            col = POS.index(pos1)
            row = POS.index(pos2)
            t[row][col]+=1
    the_sum=np.sum(t,axis=0)
    t = np.divide(t,the_sum,where=the_sum!=0)

    o=dict()
    for sentence in data:
        for item in sentence:
            word, pos = item
            idx = POS.index(pos)
            if word not in o.keys():
                o[word]=np.array([0]*len(POS))
            o[word][idx]+=1
    sum_x=np.array([0]*len(POS))
    for value in o.values():
        sum_x=np.add(sum_x,value)
    for key, value in o.items():
        o[key]=np.divide(value,sum_x,where=sum_x!=0)

    return x0, t, o


"""
3.2: Viterbi forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: m, 1D array; pointers, 2D array
"""
def viterbi_forward(X0:npt.NDArray,
                    Tprob:npt.NDArray,
                    Oprob:Dict[str,npt.NDArray],
                    obs:List[str]
                    ) -> Tuple[npt.NDArray, npt.NDArray]:
    m,t=X0,Tprob
    all_p=np.empty((0,len(POS)))
    for i in range(len(obs)):
        m_p=np.max(np.multiply(t,m),axis=1)
        p=np.argmax(np.multiply(t,m),axis=1)
        if obs[i] not in Oprob.keys():
            o=np.array([1]*len(POS))
        else:
            o=Oprob[obs[i]]
        m=np.multiply(o,m_p)
        all_p=np.append(all_p,[p],axis=0)

    return m, all_p

"""
3.2: Viterbi backward algorithm
Param: m, 1D array; pointers, 2D array
Return: List of most likely POS (strings)
"""
def viterbi_backward(m:npt.NDArray,
                     pointers:npt.NDArray
                     ) -> List[str]:
    states_idx=[]
    p=np.argmax(m)
    states_idx.append(int(p))
    for row in np.flip(pointers,axis=0):
        p=row[int(p)]
        states_idx.insert(0,int(p))
    states_idx.pop(0)
    states=[POS[i] for i in states_idx]

    return states


"""
3.3: Evaluate Viterbi by predicting on data set and returning accuracy rate
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of (word,POS) pairs
Return: Prediction accuracy rate
"""
def evaluate_viterbi(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     data:List[List[Tuple[str]]]
                     ) -> float:
    total_acc=0
    for sentence in data:
        words,true_pos = [],[]
        for word, pos in sentence:
            words.append(word)
            true_pos.append(pos)
        m, pointers = viterbi_forward(X0,Tprob,Oprob,words)
        predicted_pos=np.array(viterbi_backward(m,pointers))
        true_pos=np.array(true_pos)
        accuracy = np.mean(predicted_pos==true_pos)
        total_acc+=accuracy
    total_acc/=len(data)
    return total_acc


"""
3.4: Forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: P(XT, e_1:T)
"""
def forward(X0:npt.NDArray,
            Tprob:npt.NDArray,
            Oprob:Dict[str,npt.NDArray],
            obs:List[str]
            ) -> npt.NDArray:
    a,t=np.expand_dims(X0,axis=1),Tprob
    for i in range(len(obs)):
        a_p=np.dot(t,a)
        if obs[i] not in Oprob.keys():
            o = np.array([1] * len(POS))
        else:
            o = Oprob[obs[i]]
        o=np.expand_dims(o,axis=1)
        a=np.multiply(o,a_p)
    a=np.squeeze(a,axis=1)

    return a

"""
3.4: Backward algorithm
Param: Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(e_k+1:T | Xk)
"""
def backward(Tprob:npt.NDArray,
             Oprob:Dict[str,npt.NDArray],
             obs:List[str],
             k:int
             ) -> npt.NDArray:
    t=Tprob
    b=np.array([1] * len(POS))
    b=np.expand_dims(b,axis=1)
    obs.reverse()
    for i in range(len(obs)-k-1):
        if obs[i] not in Oprob.keys():
            o = np.array([1] * len(POS))
        else:
            o = Oprob[obs[i]]
        o=np.expand_dims(o,axis=1)
        b_p=np.multiply(o,b)
        b=t.T@b_p
    b=np.squeeze(b,axis=1)

    return b

"""
3.4: Forward-backward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(Xk | e_1:T)
"""
def forward_backward(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     obs:List[str],
                     k:int
                     ) -> npt.NDArray:
    x0,t=X0,Tprob
    f_obs=obs[:k+1]
    a=forward(x0,t,Oprob,f_obs)
    b=backward(t,Oprob,obs,k)
    a_b = np.multiply(a,b)
    y=np.divide(a_b,np.sum(a_b))

    return y


"""
3.5: Expected observation probabilities given data sequence
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of words
Return: New Oprob, dictionary {word:probabilities}
"""
def expected_emissions(X0:npt.NDArray,
                       Tprob:npt.NDArray,
                       Oprob:Dict[str,npt.NDArray],
                       data:List[List[str]]
                       ) -> Dict[str,npt.NDArray]:
    x0,t=X0,Tprob
    o={}
    for sentence in data:
        sum_y=np.array([0]*len(POS))
        dict_y={}
        for k in range(len(sentence)):
            word=sentence[k]
            y=forward_backward(x0,t,Oprob,sentence,k)
            if word not in dict_y.keys():
                dict_y[word]=[]
            dict_y[word].append(y)
            sum_y=np.add(sum_y,y)
        for word in dict_y.keys():
            p=np.divide(np.sum(dict_y[word],axis=0),sum_y)
            if word not in o.keys():
                o[word]=np.array([0]*len(POS))
            o[word]=np.add(o[word],p)
    sum_x = np.array([0] * len(POS))
    for value in o.values():
        sum_x=np.add(sum_x,value)
    for key, value in o.items():
        o[key]=np.divide(value,sum_x)

    return o


if __name__ == "__main__":
    # Run below for 3.3
    train = read_corpus('train.upos.tsv')
    test = read_corpus('test.upos.tsv')
    X0, T, O = learn_model(train)
    print("Train accuracy:", evaluate_viterbi(X0, T, O, train))
    print("Test accuracy:", evaluate_viterbi(X0, T, O, test))

    # Run below for 3.5
    obs = [[pair[0] for pair in sentence] for sentence in [test[0]]]
    # Onew = expected_emissions(X0, T, O, [['What', 'if', 'Google', 'Morphed', 'Into', 'GoogleOS', '?'],
    #                                      ['[', 'via', 'Microsoft', 'Watch', 'from', 'Mary', 'Jo', 'Foley', ']']])
    Onew = expected_emissions(X0, T, O, obs)
    print(Onew)
