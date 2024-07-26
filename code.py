import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WordSequencer:
    def init(self):
        self.model = None
        self.bigram_indices = None

    def fit(self, dictionary):
        """
        Train the model using the provided dictionary.
        :param dictionary: List of words to train on.
        """
        # Extract all unique bigrams from the dictionary
        bigrams = set()
        for word in dictionary:
            for i in range(len(word) - 1):
                bigrams.add(word[i:i + 2])
        
        bigram_list = sorted(bigrams)
        self.bigram_indices = {bigram: idx for idx, bigram in enumerate(bigram_list)}

        # Create feature vectors
        X = np.zeros((len(dictionary), len(bigram_list)))
        y = np.array(dictionary)

        for i, word in enumerate(dictionary):
            for j in range(len(word) - 1):
                bigram = word[j:j + 2]
                if bigram in self.bigram_indices:
                    X[i, self.bigram_indices[bigram]] = 1

        # Train a decision tree classifier
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_split=5)
        self.model.fit(X, y)
    
    def predict(self, bigrams):
        """
        Predict the word(s) given a list of bigrams.
        :param bigrams: Tuple of bigrams sorted in lexicographic order.
        :return: List of predicted words.
        """
        # Create feature vector for the input bigrams
        feature_vector = np.zeros((1, len(self.bigram_indices)))

        for bigram in bigrams:
            if bigram in self.bigram_indices:
                feature_vector[0, self.bigram_indices[bigram]] = 1

        # Predict the probabilities
        probabilities = self.model.predict_proba(feature_vector)[0]
        sorted_indices = np.argsort(probabilities)[::-1]

        # Retrieve up to 5 guesses
        possible_words = [self.model.classes_[idx] for idx in sorted_indices]
        guesses = []
        for word in possible_words:
            if all(bigram in word for bigram in bigrams):
                guesses.append(word)
            if len(guesses) == 5:
                break

        return guesses

# Global instance of WordSequencer
word_sequencer = WordSequencer()

################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################

    # Train the model using the word list provided
    word_sequencer.fit(words)
    return word_sequencer


################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################

    # Predict on a test bigram_list
    guesses = model.predict(bigram_list)
    return guesses  # Return guess(es) as a list