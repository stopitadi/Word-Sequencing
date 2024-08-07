## Motivation

In the field of genetics, sequencing a genome is a challenging yet crucial task that involves determining the exact sequence of base pairs in an organism's DNA. However, direct sequencing is often not possible because DNA is usually available only in short strands rather than as a single long chain. By sequencing these short strands and reconstructing the genome using overlapping sequences, we can achieve a complete genome sequence. This problem has fascinating parallels in other fields, such as natural language processing (NLP), where reconstructing original data from partial information is a common challenge. Inspired by the exciting applications of genetics, such as synthetic biology and CRISPR, we aim to tackle a similar problem in NLP.

## Task Description

Our task involves reconstructing English words given a set of bigrams (pairs of consecutive letters) that appear in those words. The challenge is that these bigrams are provided in a sorted and deduplicated manner, and only the first five bigrams are retained. Due to this preprocessing, different words can result in identical bigram sets, causing ambiguity. For example, the words "optional" and "proportional" may both result in the bigram set ('al', 'io', 'na', 'on', 'op') due to the constraints on the number of bigrams.

Given a list of bigrams as a Python tuple, sorted lexicographically and with at most five elements, our task is to guess one or more possible words that correspond to that list. We can make up to five guesses, and our precision score is determined by the number of correct guesses among the total guesses made.

## Model Design

To address this problem, we have designed a Decision Tree Model. This model helps us to systematically analyze the provided bigram sets and predict the possible corresponding words. The decision tree approach allows us to handle the ambiguities and potential clashes arising from the preprocessing steps, ensuring that our model can efficiently map bigram sets to their possible word origins.
