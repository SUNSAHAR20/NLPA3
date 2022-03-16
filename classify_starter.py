import re, math
from collections import defaultdict

# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        return self.mfc


# A most-frequent class baseline
class PolarityBaseline:
    def __init__(self):
        result = ''

    def count(self, t, ptokens, ntokens):
        # Count classes to determine which is the most frequent
            pos_words_set = []
            neg_words_set = []
            pos_word_freqs = 0
            neg_word_freqs = 0
            tokens = tokenize(t)
            for pos in ptokens:
                if pos in tokens:
                    pos_words_set.append(pos)
                    pos_word_freqs += 1
            for neg in ntokens:
                if neg in tokens:
                    neg_words_set.append(neg)
                    neg_word_freqs += 1

            if(pos_word_freqs > neg_word_freqs):
                result = 'positive'
            elif (pos_word_freqs < neg_word_freqs):
                result = 'negative'
            else:
                result = 'neutral'
            return result

class NaiveBayes2:
    def __init__(self, klasses, train_texts):
        self.train(klasses, train_texts)

    def train(self, klasses, train_texts):
        # Count classes for prior probabilities
        self.pos_likelihood_log = {}
        self.neg_likelihood_log = {}
        self.neu_likelihood_log = {}
        klass_freqs = {}
        self.priorlog = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.total_klasses = len(klass_freqs)
        self.priorlog['positive'] = math.log(klass_freqs['positive'] / self.total_klasses)
        self.priorlog['negative'] = math.log(klass_freqs['negative'] / self.total_klasses)
        self.priorlog['neutral'] = math.log(klass_freqs['neutral'] / self.total_klasses)

        #Maximum likelihood probability for training docs
        word_in_pos_doc_freqs = {}
        word_in_neg_doc_freqs = {}
        word_in_neu_doc_freqs = {}
        self.word_vocabulary_freqs = {}
        pos_word_tokens = 0
        neg_word_tokens = 0
        neu_word_tokens = 0
        for text, klass in zip(train_texts, klasses):
            tokens = tokenize(text)
            for t in tokens:
                if klass == "positive":
                    word_in_pos_doc_freqs[t] = word_in_pos_doc_freqs.get(t, 0) + 1
                    pos_word_tokens += 1
                if klass == "negative":
                    word_in_neg_doc_freqs[t] = word_in_neg_doc_freqs.get(t, 0) + 1
                    neg_word_tokens += 1
                if klass == "neutral":
                    word_in_neu_doc_freqs[t] = word_in_neu_doc_freqs.get(t, 0) + 1
                    neu_word_tokens += 1
                self.word_vocabulary_freqs[t] = self.word_vocabulary_freqs.get(t, 0) + 1

        for word in self.word_vocabulary_freqs.keys():
            if word in word_in_pos_doc_freqs:
                self.pos_likelihood_log[word] = self.pos_likelihood_log.get(word, 0) + math.log((word_in_pos_doc_freqs[word] + 1) / (pos_word_tokens + len(self.word_vocabulary_freqs)))
            if word in word_in_neg_doc_freqs:
                self.neg_likelihood_log[word] = self.neg_likelihood_log.get(word, 0) + math.log((word_in_neg_doc_freqs[word] + 1) / (neg_word_tokens + len(self.word_vocabulary_freqs)))
            if word in word_in_neu_doc_freqs:
                self.neu_likelihood_log[word] = self.neu_likelihood_log.get(word, 0) + math.log((word_in_neu_doc_freqs[word] + 1) / (neu_word_tokens + len(self.word_vocabulary_freqs)))

    def classify(self, text):
        sum = {}
        test_tokens = tokenize(text)
        for klass in self.priorlog.keys():
            sum[klass] = self.priorlog[klass]
            for test in test_tokens:
                if test in self.word_vocabulary_freqs.keys():
                    if klass == "positive" and test in self.pos_likelihood_log.keys():
                        sum[klass] += self.pos_likelihood_log[test]
                    if klass == "negative" and test in self.neg_likelihood_log.keys():
                        sum[klass] += self.neg_likelihood_log[test]
                    if klass == "neutral" and test in self.neu_likelihood_log.keys():
                        sum[klass] += self.neu_likelihood_log[test]

        argmax_klass = max(sum, key=sum.get)
        return argmax_klass

class NaiveBayes:
    def __init__(self, train_klasses, train_texts):
        self.train(train_klasses, train_texts)

    def train(self, train_klasses, train_texts):
        self.likelihood_log = {}
        bigdoc_words_per_class_freqs = defaultdict(dict)
        self.priorlog = {}
        self.word_vocabulary_freqs = {}
        klass_freqs = {}
        word_tokens = {}
        bigdoc = defaultdict(list)

        for text, klass in zip(train_texts, train_klasses):
            klass_freqs[klass] = klass_freqs.get(klass, 0) + 1
            bigdoc[klass].append(text)
            tokens = tokenize(text)
            for t in tokens:
                self.word_vocabulary_freqs[t] = self.word_vocabulary_freqs.get(t, 0) + 1

        # Count classes for prior probabilities
        self.total_klasses = len(train_klasses)
        for k in klass_freqs.keys():
            self.priorlog[k] = abs(math.log(klass_freqs[k]) - math.log(self.total_klasses))

            for b in bigdoc[k]:
                bigdoc_tokens = tokenize(b)
                for bt in bigdoc_tokens:
                    if bt in self.word_vocabulary_freqs.keys():
                        word_tokens[k] = word_tokens.get(k, 0) + 1
                        bigdoc_words_per_class_freqs[k][bt] = bigdoc_words_per_class_freqs[k].get(bt, 0) + 1

        #Maximum likelihood probability for training docs
        for klass_label in self.priorlog.keys():
            for word in self.word_vocabulary_freqs.keys():
                if word in bigdoc_words_per_class_freqs[klass_label]:
                    self.likelihood_log[(word, klass_label)] = abs(math.log((bigdoc_words_per_class_freqs[klass_label][word] + 1)) - math.log((word_tokens[klass_label] + len(self.word_vocabulary_freqs))))

    def classify(self, text):
        sum = {}
        test_tokens = tokenize(text)
        for sklass in self.priorlog.keys():
            sum[sklass] = self.priorlog[sklass]
            for test in test_tokens:
                if test in self.word_vocabulary_freqs.keys():
                    key_check = (test, sklass)
                    if key_check in self.likelihood_log:
                        sum[sklass] += self.likelihood_log[(test, sklass)]

        argmax_klass = max(sum, key=sum.get)
        return argmax_klass

class BinaryNaiveBayes:
    def __init__(self, train_klasses, train_texts):
        self.train(train_klasses, train_texts)

    def train(self, train_klasses, train_texts):
        self.likelihood_log = {}
        bigdoc_words_per_class_freqs = defaultdict(dict)
        self.priorlog = {}
        self.word_vocabulary_freqs = {}
        klass_freqs = {}
        word_tokens = {}
        bigdoc = defaultdict(list)

        for text, klass in zip(train_texts, train_klasses):
            klass_freqs[klass] = klass_freqs.get(klass, 0) + 1
            bigdoc[klass].append(text)
            tokens = set(tokenize(text))
            for t in tokens:
                self.word_vocabulary_freqs[t] = self.word_vocabulary_freqs.get(t, 0) + 1

        # Count classes for prior probabilities
        self.total_klasses = len(train_klasses)
        for k in klass_freqs.keys():
            self.priorlog[k] = abs(math.log(klass_freqs[k]) - math.log(self.total_klasses))

            for b in bigdoc[k]:
                bigdoc_tokens = set(tokenize(b))  #ignoring repeating tokens pre-binarization
                for bt in bigdoc_tokens:
                    if bt in self.word_vocabulary_freqs.keys():
                        word_tokens[k] = word_tokens.get(k, 0) + 1
                        bigdoc_words_per_class_freqs[k][bt] = bigdoc_words_per_class_freqs[k].get(bt, 0) + 1


        #Maximum likelihood probability for training docs
        for klass_label in self.priorlog.keys():
            for word in self.word_vocabulary_freqs.keys():
                if word in bigdoc_words_per_class_freqs[klass_label]:
                    self.likelihood_log[(word, klass_label)] = abs(math.log((bigdoc_words_per_class_freqs[klass_label][word] + 1)) - math.log((word_tokens[klass_label] + len(self.word_vocabulary_freqs))))

    def classify(self, text):
        sum = {}
        test_tokens = set(tokenize(text))
        for sklass in self.priorlog.keys():
            sum[sklass] = self.priorlog[sklass]
            for test in test_tokens:
                if test in self.word_vocabulary_freqs.keys():
                    key_check = (test, sklass)
                    if key_check in self.likelihood_log:
                        sum[sklass] += self.likelihood_log[(test, sklass)]

        argmax_klass = max(sum, key=sum.get)
        return argmax_klass

if __name__ == '__main__':
    import sys
    #sys.stdout.reconfigure(encoding='utf-8')
    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or 'nbbin'
    method = sys.argv[1]
    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]
    
    train_texts = [x.strip() for x in open(train_texts_fname,
                                         encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                          encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]
    pos_tokens = [x.strip() for x in open('pos-words.txt',
                                          encoding='utf8')]
    neg_tokens = [x.strip() for x in open('neg-words.txt',
                                          encoding='utf8')]

    if method == 'lexicon':
        polarity_classifier = PolarityBaseline()
        results = [polarity_classifier.count(text, pos_tokens, neg_tokens) for text in test_texts]

    elif method == 'nb':
        nb_classifier = NaiveBayes(train_klasses, train_texts)
        results = [nb_classifier.classify(x) for x in test_texts]

    elif method == 'nbbin':
        nbbin_classifier = BinaryNaiveBayes(train_klasses, train_texts)
        results = [nbbin_classifier.classify(x) for x in test_texts]

    elif method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        train_counts = count_vectorizer.fit_transform(train_texts)

        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        test_counts = count_vectorizer.transform(test_texts)
        # Predict the class for each test document
        results = clf.predict(test_counts)

    for r in results:
        print(r)
