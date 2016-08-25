import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
#from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    
    import itertools
    data = train_neg + train_pos
    all_words = set(itertools.chain(*data))
    all_words = all_words - stopwords           #Remove all stop words

    pos = [set(x) for x in train_pos]
    neg = [set(x) for x in train_neg]
    

    #cnt1 is the dictionary which keeps the frequency count of each positive word
    #cnt2 is the dictionary which keeps the frequency count of each negative word
    
    cnt1 = {}
    cnt2 = {}
    #Flattening the positive and negative arrays
    pos_ = list(itertools.chain(*pos))
    neg_ = list(itertools.chain(*neg))
    
    for x in pos_:
        try:
            if(x in all_words):
                cnt1[x]+=1
        except KeyError:
            cnt1[x]=1

    for x in neg_:
        try:
            if(x in all_words):
                cnt2[x]+=1
        except KeyError:
            cnt2[x]=1


    
    merged = set(pos_ + neg_)
    filtered = []
    for word in merged:
        if((word in cnt1 and cnt1[word]*100>=len(pos)) or (word in cnt2 and cnt2[word]*100 >= len(neg))):
            filtered += [word]

    ans = []
    for word in filtered:
        if(word in cnt1 and word in cnt2 and (cnt1[word] >= 2*cnt2[word] or cnt2[word] >= 2*cnt1[word])):
            ans += [word]

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE    
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for x in train_pos:
        tst = set(x)
        lst = [0]*len(ans)
        for i in xrange(len(ans)):
            lst[i] = int(ans[i] in tst)
        train_pos_vec.append(lst)
  
    for x in train_neg:
        tst = set(x)
        lst = [0]*len(ans)
        for i in xrange(len(ans)):
            lst[i] = int(ans[i] in tst)
        train_neg_vec.append(lst)


    for x in test_pos:
        tst = set(x)
        lst = [0]*len(ans)
        for i in xrange(len(ans)):
            lst[i] = int(ans[i] in tst)
        test_pos_vec.append(lst)


    for x in test_neg:
        tst = set(x)
        lst = [0]*len(ans)
        for i in xrange(len(ans)):
            lst[i] = int(ans[i] in tst)
        test_neg_vec.append(lst)

    print len(test_neg_vec[0])
    print test_neg_vec[0]
   # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec




def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []


    for i,lst in enumerate(train_pos):
        lst = train_pos[i]
        labeled_train_pos += [LabeledSentence(words=lst, tags=["TRAIN_POS_"+str(i)])]

    for i,lst in enumerate(train_neg):
        lst = train_neg[i]
        labeled_train_neg += [LabeledSentence(words=lst, tags=["TRAIN_NEG_"+str(i)])]


    for i,lst in enumerate(test_pos):
        lst = test_pos[i]
        
        labeled_test_pos += [LabeledSentence(words=lst, tags=["TEST_POS_"+str(i)])]

    for i,lst in enumerate(test_neg):
        lst = test_neg[i]
        
        labeled_test_neg += [LabeledSentence(words=lst, tags=["TEST_NEG_"+str(i)])]

    # Initialize model

    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    
    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = []
    test_pos_vec = []
    train_neg_vec = []
    test_neg_vec = []

    for i in xrange(len(train_pos)): 
        train_pos_vec.append(model.docvecs["TRAIN_POS_"+str(i)])
    for i in xrange(len(train_neg)):
        train_neg_vec.append(model.docvecs["TRAIN_NEG_"+str(i)])
    for i in xrange(len(test_pos)):
        test_pos_vec.append(model.docvecs["TEST_POS_"+str(i)])
    for i in xrange(len(test_neg)):
        test_neg_vec.append(model.docvecs["TEST_NEG_"+str(i)])

    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    
    
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb = sklearn.naive_bayes.BernoulliNB(alpha=0.1,binarize = None)
    lr = sklearn.linear_model.LogisticRegression()
    nb_model = nb.fit(X,Y)
    lr_model = lr.fit(X,Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    
    
    nb = sklearn.naive_bayes.GaussianNB()
    lr = sklearn.linear_model.LogisticRegression()
    nb_model = nb.fit(X,Y)
    lr_model = lr.fit(X,Y)
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    
    expected = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)
    predicted = model.predict(test_pos_vec + test_neg_vec)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in xrange(len(expected)):
        if(expected[i]=='pos' and predicted[i]=='pos'):
            tp+=1
        if(expected[i]=='neg' and predicted[i]=='neg'):
            tn+=1
        if(expected[i]=='pos' and predicted[i]=='neg'):
            fn+=1
        if(expected[i]=='neg' and predicted[i]=='pos'):
            fp+=1
    accuracy = (tp + tn)*1.0/(tp + tn + fn + fp)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
        print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
