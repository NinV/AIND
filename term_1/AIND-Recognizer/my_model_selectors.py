import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        n_states_to_score = {}
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_states)
                score = -2 * model.score(self.X, self.lengths)

                # number of free parameters: n_params = n_initial_prob + n_transition_prob + n_emission
                # where:
                #   n_initial_prob = n_states-1
                #   n_transition = n_states * (n_states-1)
                #   n_emission = 2 * n_feature * n_states
                n_params = (n_states-1) + n_states*(n_states-1) + 2 * self.X.shape[1] * n_states

                score += n_params * math.log(self.X.shape[0], 10)
                n_states_to_score[n_states] = score
            except:
                continue

        if n_states_to_score:
            best_n_states = max(n_states_to_score.keys(), key= lambda k: n_states_to_score[k])
            return self.base_model(best_n_states)
        return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        n_states_to_score = {}
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_states)

                # compute the likelyhood/evidence of the model for training word
                evidence = model.score(self.X, self.lengths)

                # compute the anti-evidence of the model for training word
                anti_evidence = 0
                competing_word_set = set(self.words.keys())
                competing_word_set.discard(self.this_word)
                for competing_word in competing_word_set:
                    X, lengths = self.hwords[competing_word]
                    anti_evidence += model.score(X, lengths)
                anti_evidence /= (len(self.words.keys())-1)
                n_states_to_score[n_states] = evidence-anti_evidence
            except:
                continue

        if n_states_to_score:
            best_n_states = max(n_states_to_score.keys(), key= lambda k: n_states_to_score[k])
            return self.base_model(best_n_states)
        return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        split_method = KFold()
        logL_score = {}
        for n_states in range(self.min_n_components, self.max_n_components+1):

            # if the word has number of samples less than number of folds then we will train
            # the model with the whole dataset
            if len(self.sequences) < split_method.get_n_splits():
                try:
                    model = self.base_model(n_states)
                    logL_score[n_states] = model.score(self.X, self.lengths)
                except:
                    continue

            # using k-fold cross-validation when we have enough samples
            else:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    logL_score[n_states] = 0

                    # create train sequences and test sequences from train_idx and test_idx
                    test_sequences_X, test_sequences_lenghts = combine_sequences(cv_test_idx, self.sequences)
                    train_sequences_X, train_sequences_lenghts = combine_sequences(cv_train_idx, self.sequences)

                    # create model, train and test
                    try:
                        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(train_sequences_X, train_sequences_lenghts)
                        logL = model.score(test_sequences_X, test_sequences_lenghts)
                        logL_score[n_states] += logL / split_method.get_n_splits()
                    except:
                        break

        # the best model have the highest logL
        if logL_score:
            best_n_states = max(logL_score.keys(), key= lambda k: logL_score[k])
            return self.base_model(best_n_states)
        return self.base_model(self.n_constant)
