"""
===========================
Functions for each individual operation mode
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

from os import path

from numpy import nan
from pandas import DataFrame, read_csv

from ldm.corpus.corpus import CorpusMetadata
from ldm.corpus.indexing import FreqDist
from ldm.corpus.multiword import VectorCombinatorType
from ldm.utils.exceptions import WordNotFoundError
from ldm.utils.maths import DistanceType

FIRST_WORD = "First word"
SECOND_WORD = "Second word"


def _frequency(word: str, freq_dist: FreqDist) -> int:
    try:
        return freq_dist[word]
    except KeyError:
        return 0


def run_frequency(word: str,
                  freq_dist: FreqDist,
                  output_file: str):
    occurrences = _frequency(word, freq_dist)
    if output_file is None:
        print(occurrences)
    else:
        with open(output_file, mode="w", encoding="utf-8") as f:
            f.write(f"{occurrences}\n")


def run_frequency_with_list(wordlist_file: str,
                            freq_dist: FreqDist,
                            corpus: CorpusMetadata,
                            output_file: str):
    with open(wordlist_file, mode="r") as wf:
        word_list = [l.strip().lower() for l in wf]

    freqs = []
    for word in word_list:
        freqs.append((word, _frequency(word, freq_dist)))

    if output_file is None:
        for entry in freqs:
            print(f"{entry[0]}: {entry[1]}")
    else:
        (DataFrame
         .from_records(freqs,
                       columns=["Word", f"Frequency in {corpus.name} corpus"])
         .to_csv(output_file, header=True, index=False))


def _rank(word: str, freq_dist: FreqDist) -> int:
    r = freq_dist.rank(word)
    # +1 means that the most-frequent word is 1
    # freq_dist.rank returns -1 if the word is not found, meaning that
    # this function will return 0 if the word is not found.
    # so use >= 1 checks for if the word is found
    return r + 1


def run_rank(word: str,
             freq_dist: FreqDist,
             output_file: str):
    rank = _rank(word, freq_dist)

    if output_file is None:
        print(rank if rank >= 1 else "None")
    else:
        with open(output_file, mode="w", encoding="utf-8") as f:
            if rank >= 1:
                f.write(f"{rank}\n")
            else:
                f.write("None\n")


def run_rank_with_list(wordlist_file: str,
                       freq_dist: FreqDist,
                       corpus: CorpusMetadata,
                       output_file: str):
    with open(wordlist_file, mode="r") as wf:
        word_list = [l.strip().lower() for l in wf]

    ranks = []
    for word in word_list:
        ranks.append((word, _rank(word, freq_dist)))

    if output_file is None:
        for word, rank in ranks:
            if rank >= 1:
                print(f"{word}: {rank}")
            else:
                print(f"{word}: None")
    else:
        rank_col_name = f"Frequency in {corpus.name} corpus"
        data = DataFrame.from_records(ranks, columns=["Word", rank_col_name])
        data[data[rank_col_name] == -1] = nan
        data.to_csv(output_file, header=True, index=False)


def run_vector(word: str,
               model,
               output_file: str):
    load_model(model)
    try:
        vector = model.vector_for_word(word)

        if output_file is None:
            print(vector)
        else:
            with open(output_file, mode="w", encoding="utf-8") as f:
                f.write(f"{vector}\n")
    except WordNotFoundError:
        print(f"{word} not found in {model.corpus_meta.name} corpus")


def run_vector_with_list(wordlist_file: str,
                         model,
                         output_file: str):
    load_model(model)

    with open(wordlist_file, mode="r") as wf:
        word_list = [l.strip().lower() for l in wf]

    if output_file is None:
        missing_words = []
        for word in word_list:
            try:
                vector = model.vector_for_word(word)
                print(f"{word}: {vector}")
            except WordNotFoundError:
                missing_words.append(word)
        if len(missing_words) > 0:
            print(f"The following words were not found in the {model.corpus_meta.name} corpus")
            print(", ".join(missing_words))


def _compare(word_1, word_2, model, distance: DistanceType, combinator_type: VectorCombinatorType) -> float:

    from ldm.model.ngram import NgramModel
    from ldm.model.base import VectorModel

    try:
        if isinstance(model, NgramModel):
            return model.association_between(word_1, word_2)
        elif isinstance(model, VectorModel):
            if combinator_type is VectorCombinatorType.none:
                return model.distance_between(word_1, word_2, distance)
            elif combinator_type in [VectorCombinatorType.additive,
                                     VectorCombinatorType.multiplicative,
                                     VectorCombinatorType.mean]:
                return model.distance_between_multigrams(word_1, word_2, distance, combinator_type)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    except WordNotFoundError:
        return nan


def run_compare(word_1: str, word_2: str,
                model,
                distance: DistanceType,
                combinator_type: VectorCombinatorType,
                output_file: str):
    load_model(model)

    comparison = _compare(word_1, word_2, model, distance, combinator_type)

    if output_file is None:
        print(comparison)
    else:
        with open(output_file, mode="r", encoding="utf-8") as f:
            f.write(f"{comparison}\n")


def run_compare_with_list(wordlist_file: str,
                          model,
                          distance: DistanceType,
                          combinator_type: VectorCombinatorType,
                          output_file: str):
    load_model(model)

    with open(wordlist_file, mode="r", encoding="utf-8") as wf:
        word_list = [l.strip().lower() for l in wf]

    # Do it the slow and obvious way
    matrix = []
    for word_1 in word_list:
        # First column will be the first word
        # noinspection PyTypeChecker
        row = (word_1,) + tuple(
            # Remaining columns will be comparison with each word in the list
            _compare(word_1, word_2, model, distance, combinator_type)
            for word_2 in word_list)
        matrix.append(row)

    data = DataFrame.from_records(matrix, columns=[FIRST_WORD] + word_list).set_index(FIRST_WORD)

    if output_file is None:
        for line in str(data):
            print(line)
    else:
        data.to_csv(output_file, header=True, index=True)


def run_compare_with_pair_list(wordpair_list_file: str,
                               model,
                               distance: DistanceType,
                               combinator_type: VectorCombinatorType,
                               output_file: str):
    load_model(model)

    with open(wordpair_list_file, mode="r", encoding="utf-8") as wf:
        wordpair_list_df = read_csv(wf, header=None, index_col=None,
                                    names=[FIRST_WORD, SECOND_WORD])

    if output_file is None:
        for _, word_1, word_2 in wordpair_list_df.itertuples():
            comparison = _compare(word_1, word_2, model, distance, combinator_type)
            print(f"({word_1}, {word_2}): {comparison}")
    else:
        if distance is not None:
            comparison_col_name = f"{distance.name} distance"
        else:
            comparison_col_name = "Association"

        if combinator_type is not VectorCombinatorType.none:
            comparison_col_name += f" ({combinator_type.name})"

        rows = []
        for i, (_, word_1, word_2) in enumerate(wordpair_list_df.itertuples(), 1):
            if i % 1000 == 0:
                print(f"Compared {i:,} pairs...")
            comparison = _compare(word_1, word_2, model, distance, combinator_type)
            rows.append((word_1, word_2, comparison))
        DataFrame.from_records(
            rows,
            columns=[FIRST_WORD, SECOND_WORD, comparison_col_name]
        ).to_csv(output_file, header=True, index=False)


def load_model(model):
    """
    Loads a model if it can.
    :param model:
    :return:
    :raises: PrecomputedModelNotFoundError
    """
    if not model.could_load:
        raise PrecomputedModelNotFoundError(f"Precomputed model not found (looking for {path.join(model.save_dir, model._model_filename_with_ext)})")
    model.train(memory_map=True)


class PrecomputedModelNotFoundError(FileNotFoundError):
    """When a precomputed model cannot be found but is required."""
    pass
