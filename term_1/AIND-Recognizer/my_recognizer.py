import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    all_Xlength = test_set.get_all_Xlengths()
    for sequence in test_set.get_all_sequences():
        word_to_score = {}
        X, lengths = all_Xlength[sequence]
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except:
                logL = float("-inf")
            word_to_score[word] = logL
        best_guesses = max(word_to_score.keys(), key= lambda k: word_to_score[k])
        probabilities.append(word_to_score)
        guesses.append(best_guesses)

    return probabilities, guesses



