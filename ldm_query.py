"""
===========================
Query corpora and linguistic distributional models.
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

import argparse
import logging
import sys
from enum import Enum, auto
from os import path

from ldm.corpus.corpus import CorpusMetadata
from ldm.corpus.indexing import FreqDist
from ldm.utils.maths import DistanceType
from ldm.corpus.multiword import VectorCombinatorType
from ldm.preferences.config import Config as LDMConfig
from operation import run_frequency, run_frequency_with_list, run_rank, run_rank_with_list, run_vector, \
    run_vector_with_list, run_compare, run_compare_with_list, run_compare_with_pair_list


# Suppress logging
logger = logging.getLogger('my-logger')
logger.propagate = False

# shortname → dirname
_corpora = {
    "bnc": "BNC",
    "subtitles": "BBC",
    "ukwac": "UKWAC",
}

_ngram_models = [
    "log-ngram",
    "conditional-probability-ngram",
    "probability-ratio-ngram",
    "ppmi-ngram",
]
_count_models = [
    "log-cooccurrence",
    "conditional-probability",
    "probability-ratio",
    "ppmi",
]
_predict_models = [
    "skip-gram",
    "cbow",
]
_models = _ngram_models + _count_models + _predict_models

_embedding_sizes = [50, 100, 200, 300, 500]
_window_radii = [1, 3, 5, 10]

_readme_path = path.join(path.dirname(path.realpath(__file__)), 'README.md')
_config_path = path.join(path.dirname(path.realpath(__file__)), 'config.yaml')


class Mode(Enum):
    """The main invocation mode of the program."""
    Frequency = auto()
    Rank = auto()
    Vector = auto()
    Compare = auto()

    @property
    def name(self) -> str:
        if self is Mode.Frequency:
            return "frequency"
        elif self is Mode.Rank:
            return "rank"
        elif self is Mode.Vector:
            return "vector"
        elif self is Mode.Compare:
            return "compare"
        else:
            raise NotImplementedError()

    @property
    def option(self) -> str:
        return '--' + self.name


class WordMode(Enum):
    """How words will be supplied"""
    # One word from CLI
    SingleWord = auto()
    # List of words from file
    SingleWordList = auto()
    # Word pair from CLI
    WordPair = auto()
    # List of word pairs from file
    WordPairList = auto()


def main(ldm_config: LDMConfig):

    argparser = build_argparser()

    args = argparser.parse_args()

    def _option_used(option_name):
        if option_name in vars(args):
            if vars(args)[option_name]:
                return True
            else:
                return False
        else:
            return False

    # region Get mode

    if args.mode == Mode.Frequency.name:
        mode = Mode.Frequency
    elif args.mode == Mode.Rank.name:
        mode = Mode.Rank
    elif args.mode == Mode.Vector.name:
        mode = Mode.Vector
    elif args.mode == Mode.Compare.name:
        mode = Mode.Compare
    else:
        raise NotImplementedError()

    # region Validate args

    # Validate model params
    if _option_used("model"):

        # For predict models, embedding size is required
        if args.model[0].lower() in _predict_models:
            if len(args.model) == 1:
                argparser.error("Please specify embedding size when using predict models")
            elif int(args.model[1]) not in _embedding_sizes:
                    argparser.error(f"Invalid embedding size {args.model[1]}, "
                                    f"Please select an embedding size from the list {_embedding_sizes}")

        # For count and ngram models, embedding size is forbidden
        else:
            if len(args.model) > 1:
                argparser.error("Embedding size invalid for count and n-gram models")

    # Validate vector mode
    if mode is Mode.Vector:
        if args.model[0].lower() in _ngram_models:
            argparser.error("Cannot use n-gram model in vector mode.")

    # Validate distance measure
    if mode is Mode.Compare:
        # All but n-grams require distance
        if args.model[0].lower() in _ngram_models:
            if args.distance is not None:
                argparser.error("Distance not valid for n-gram models")
        else:
            if args.distance is None:
                argparser.error("Distance is required for vector-based models.")
                
    # Validate combinator type
    if mode is Mode.Compare:
        # Combinators can only be used with vector models, but is not required
        if args.model[0].lower() in _ngram_models:
            if args.combinator is not None:
                argparser.error("Combinator not valid for n-gram models")

    # endregion

    # region Interpret args

    # Get word_mode
    # and words or path
    if _option_used("word"):
        word_mode = WordMode.SingleWord
        words_or_path = args.word
    elif _option_used("word_pair"):
        word_mode = WordMode.WordPair
        words_or_path = args.word_pair
    elif _option_used("words_from_file"):
        word_mode = WordMode.SingleWordList
        words_or_path = args.words_from_file
    elif _option_used("word_pairs_from_file"):
        word_mode = WordMode.WordPairList
        words_or_path = args.word_pairs_from_file
    else:
        raise NotImplementedError()

    # get model spec
    if not _option_used("model"):
        model_type = None
        embedding_size = None
    elif len(args.model) == 1:
        model_type = args.model[0]
        embedding_size = None
    elif len(args.model) == 2:
        model_type = args.model[0]
        embedding_size = int(args.model[1])
    else:
        raise NotImplementedError()
    radius = int(args.window_radius) if "window_radius" in vars(args) else None

    if not _option_used("distance"):
        distance = None
    elif args.distance.lower() == "cosine":
        distance = DistanceType.cosine
    elif args.distance.lower() == "correlation":
        distance = DistanceType.correlation
    elif args.distance.lower() == "euclidean":
        distance = DistanceType.Euclidean
    else:
        raise NotImplementedError()

    if not _option_used("combinator"):
        combinator_type = VectorCombinatorType.none
    elif args.combinator == VectorCombinatorType.none.name:
        combinator_type = VectorCombinatorType.none
    elif args.combinator == VectorCombinatorType.additive.name:
        combinator_type = VectorCombinatorType.additive
    elif args.combinator == VectorCombinatorType.multiplicative.name:
        combinator_type = VectorCombinatorType.multiplicative
    else:
        raise NotImplementedError()

    # Get corpus and freqdist
    corpus_name = args.corpus
    corpus: CorpusMetadata = CorpusMetadata(
        name=_corpora[corpus_name],
        path=ldm_config.value_by_key_path("corpora", corpus_name, "path"),
        freq_dist_path=ldm_config.value_by_key_path("corpora", corpus_name, "index"))
    freq_dist: FreqDist = FreqDist.load(corpus.freq_dist_path)

    # Get output file
    output_file = args.output_file

    # Build model
    model = get_model_from_parameters(model_type, radius, embedding_size, corpus, freq_dist)

    # endregion

    # region Run appropriate function based on mode

    if mode is Mode.Frequency:
        if word_mode is WordMode.SingleWord:
            run_frequency(words_or_path, freq_dist, output_file)
        elif word_mode is WordMode.SingleWordList:
            run_frequency_with_list(words_or_path, freq_dist, corpus, output_file)
        else:
            raise NotImplementedError()
    elif mode is Mode.Rank:
        if word_mode is WordMode.SingleWord:
            run_rank(words_or_path, freq_dist, output_file)
        elif word_mode is WordMode.SingleWordList:
            run_rank_with_list(words_or_path, freq_dist, corpus, output_file)
        else:
            raise NotImplementedError()
    elif mode is Mode.Vector:
        if word_mode is WordMode.SingleWord:
            run_vector(words_or_path, model, output_file)
        elif word_mode is WordMode.SingleWordList:
            run_vector_with_list(words_or_path, model, output_file)
        else:
            raise NotImplementedError()
    elif mode is Mode.Compare:
        if word_mode is WordMode.WordPair:
            run_compare(words_or_path[0], words_or_path[1], model, distance, combinator_type, output_file)
        elif word_mode is WordMode.SingleWordList:
            run_compare_with_list(words_or_path, model, distance, combinator_type, output_file)
        elif word_mode is WordMode.WordPairList:
            run_compare_with_pair_list(words_or_path, model, distance, combinator_type, output_file)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    # endregion

    sys.exit(0)


def build_argparser():
    argparser = argparse.ArgumentParser(
        description="Query corpora and linguistic distributional models. See README.md for more info.")
    # Add mode parsers
    mode_subparsers = argparser.add_subparsers(dest="mode")
    mode_subparsers.required = True
    mode_frequency_parser = mode_subparsers.add_parser(
        Mode.Frequency.name,
        help="Look up frequency of word in corpus")
    mode_rank_parser = mode_subparsers.add_parser(
        Mode.Rank.name,
        help="Look up rank of word in corpus by frequency")
    mode_vector_parser = mode_subparsers.add_parser(
        Mode.Vector.name,
        help="Look up the vector representation of a word in a model.")
    mode_compare_parser = mode_subparsers.add_parser(
        Mode.Compare.name,
        help="Compare word pairs using a model.")
    # Add corpus and outfile options to all modes
    for mode_subparser in [mode_frequency_parser, mode_rank_parser, mode_vector_parser, mode_compare_parser]:
        mode_subparser.add_argument("--corpus",
                                    type=str,
                                    choices=_corpora.keys(),
                                    required=True,
                                    help="The name of the corpus.")
        mode_subparser.add_argument("--output-file",
                                    type=str,
                                    required=False,
                                    dest="output_file",
                                    metavar="PATH",
                                    help="Write the output to this file.  Will overwrite existing files.")
    # Add single word options to relevant parsers
    for mode_subparser in [mode_frequency_parser, mode_rank_parser, mode_vector_parser]:
        wordmode_group = mode_subparser.add_mutually_exclusive_group()
        wordmode_group.add_argument("--word",
                                    type=str,
                                    required=False,
                                    help="The word to look up.")
        wordmode_group.add_argument("--words-from-file",
                                    type=str,
                                    dest="words_from_file",
                                    required=False,
                                    metavar="PATH",
                                    help="The word to look up or compare.")
    # Add all multi=word options to compare parser
    wordmode_group = mode_compare_parser.add_mutually_exclusive_group()
    wordmode_group.add_argument("--words-from-file",
                                type=str,
                                dest="words_from_file",
                                required=False,
                                metavar="PATH",
                                help="The word to look up or compare.")
    wordmode_group.add_argument("--word-pair",
                                type=str,
                                dest="word_pair",
                                nargs=2,
                                required=False,
                                metavar=('FIRST WORD', 'SECOND WORD'),
                                help="The words to compare.")
    wordmode_group.add_argument("--word-pairs-from-file",
                                type=str,
                                dest="word_pairs_from_file",
                                required=False,
                                metavar="PATH",
                                help="The word pairs to compare.")
    # Add model arguments to relevant parsers
    for mode_subparser in [mode_vector_parser, mode_compare_parser]:
        mode_subparser.add_argument("--model",
                                    type=str,
                                    choices=_models,
                                    nargs="+",
                                    required=True,
                                    dest="model",
                                    metavar=("MODEL", "EMBEDDING"),
                                    help="The model specification to use.")
        mode_subparser.add_argument("--radius",
                                    type=int,
                                    choices=_window_radii,
                                    dest="window_radius",
                                    required=True,
                                    help="The window radius to use.")
    mode_compare_parser.add_argument("--distance",
                                     type=str,
                                     choices=[dt.name for dt in DistanceType],
                                     required=False,
                                     help="The distance type to use.")
    mode_compare_parser.add_argument("--combinator",
                                     choices=[vc.name for vc in VectorCombinatorType],
                                     required=False,
                                     help="The vector combinator to use for multi-word tokens.")
    return argparser


def get_model_from_parameters(model_type: str, window_radius, embedding_size, corpus, freq_dist):
    if model_type is None:
        return None
    # Don't care about difference between underscores and hyphens
    model_type = model_type.lower().replace("_", "-")
    # N-gram models
    if model_type == "log-ngram":
        from ldm.model.ngram import LogNgramModel
        return LogNgramModel(corpus, window_radius, freq_dist)
    if model_type == "conditional-probability-ngram":
        from ldm.model.ngram import ConditionalProbabilityNgramModel
        return ConditionalProbabilityNgramModel(corpus, window_radius, freq_dist)
    if model_type == "probability-ratio-ngram":
        from ldm.model.ngram import ProbabilityRatioNgramModel
        return ProbabilityRatioNgramModel(corpus, window_radius, freq_dist)
    if model_type == "pmi-ngram":
        from ldm.model.ngram import PMINgramModel
        return PMINgramModel(corpus, window_radius, freq_dist)
    if model_type == "ppmi-ngram":
        from ldm.model.ngram import PPMINgramModel
        return PPMINgramModel(corpus, window_radius, freq_dist)
    # Count vector models:
    if model_type == "log-cooccurrence":
        from ldm.model.count import LogCoOccurrenceCountModel
        return LogCoOccurrenceCountModel(corpus, window_radius, freq_dist)
    if model_type == "conditional-probability":
        from ldm.model.count import ConditionalProbabilityModel
        return ConditionalProbabilityModel(corpus, window_radius, freq_dist)
    if model_type == "probability-ratio":
        from ldm.model.count import ProbabilityRatioModel
        return ProbabilityRatioModel(corpus, window_radius, freq_dist)
    if model_type == "pmi":
        from ldm.model.count import PMIModel
        return PMIModel(corpus, window_radius, freq_dist)
    if model_type == "ppmi":
        from ldm.model.count import PPMIModel
        return PPMIModel(corpus, window_radius, freq_dist)
    # Predict vector models:
    if model_type == "skip-gram":
        from ldm.model.predict import SkipGramModel
        return SkipGramModel(corpus, window_radius, embedding_size)
    if model_type == "cbow":
        from ldm.model.predict import CbowModel
        return CbowModel(corpus, window_radius, embedding_size)

    raise NotImplementedError()


if __name__ == '__main__':
    with LDMConfig(use_config_overrides_from_file=_config_path) as config:
        main(config)
