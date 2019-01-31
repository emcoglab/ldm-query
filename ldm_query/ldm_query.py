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

from ldm.core.corpus.corpus import CorpusMetadata
from ldm.core.corpus.indexing import FreqDist
from ldm.core.utils.maths import DistanceType
from ldm.preferences.config import Config as LDMConfig
from operation import run_frequency, run_frequency_with_list, run_rank, run_rank_with_list, run_vector, \
    run_vector_with_list, run_compare, run_compare_with_list, run_compare_with_pair_list

# Suppress logging
logger = logging.getLogger('my-logger')
logger.propagate = False

# shortname: dirname
_corpora = {
    "bnc": "BNC",
    "subtitles": "BBC",
    "ukwac": "UKWAC",
}

_models = [
    # N-gram models:
    "log-ngram",
    "probability-ratio-ngram",
    "ppmi-ngram",
    # Count vector models:
    "log-cooccurrence",
    "conditional-probability",
    "probability-ratio",
    "ppmi",
    # Predict vector models:
    "skip-gram",
    "cbow",
]

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


def main():

    # Config
    config = LDMConfig(use_config_overrides_from_file=_config_path)

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
        if args.model[0].lower() in ["cbow", "skip-gram"]:
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
        if args.model[0].lower() in ["log-ngram", "probability-ratio-ngram", "ppmi-ngram"]:
            argparser.error("Cannot use n-gram model in vector mode.")

    # Validate distance measure
    if mode is Mode.Compare:
        # All but n-grams require distance
        if args.model[0].lower() in ["log-ngram", "probability-ratio-ngram", "ppmi-ngram"]:
            if args.distance is not None:
                argparser.error("Distance not valid for n-gram models")
        else:
            if args.distance is None:
                argparser.error("Distance is required for vector-based models.")

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

    # Get corpus and freqdist
    corpus_name = args.corpus
    corpus: CorpusMetadata = CorpusMetadata(
        name=_corpora[corpus_name],
        path=config.value_by_key_path("corpora", corpus_name, "path"),
        freq_dist_path=config.value_by_key_path("corpora", corpus_name, "index"))
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
            run_compare(words_or_path[0], words_or_path[1], model, distance, output_file)
        elif word_mode is WordMode.SingleWordList:
            run_compare_with_list(words_or_path, model, distance, output_file)
        elif word_mode is WordMode.WordPairList:
            run_compare_with_pair_list(words_or_path, model, distance, output_file)
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
        help="Look up the vector representation of a model in a model.")
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
                                    nargs="+",
                                    required=True,
                                    dest="model",
                                    metavar=("MODEL", "EMBEDDING"),
                                    help="The model specification to use.")
        mode_subparser.add_argument("--window-radius",
                                    type=int,
                                    choices=_window_radii,
                                    dest="window_radius",
                                    required=True,
                                    help="The window radius to use.")
        mode_subparser.add_argument("--distance",
                                    type=str,
                                    choices=[dt.name for dt in DistanceType],
                                    required=False,
                                    help="The distance type to use.")
    return argparser


def get_model_from_parameters(model_type, window_radius, embedding_size, corpus, freq_dist):
    if model_type is None:
        model = None
    # N-gram models
    elif model_type == "log-ngram":
        from .ldm.core.model.ngram import LogNgramModel
        model = LogNgramModel(corpus, window_radius, freq_dist)
    elif model_type == "probability-ratio-ngram":
        from .ldm.core.model.ngram import ProbabilityRatioNgramModel
        model = ProbabilityRatioNgramModel(corpus, window_radius, freq_dist)
    elif model_type == "ppmi-ngram":
        from .ldm.core.model.ngram import PPMINgramModel
        model = PPMINgramModel(corpus, window_radius, freq_dist)
    # Count vector models:
    elif model_type == "log-cooccurrence":
        from .ldm.core.model.count import LogCoOccurrenceCountModel
        model = LogCoOccurrenceCountModel(corpus, window_radius, freq_dist)
    elif model_type == "conditional-probability":
        from .ldm.core.model.count import ConditionalProbabilityModel
        model = ConditionalProbabilityModel(corpus, window_radius, freq_dist)
    elif model_type == "probability-ratio":
        from .ldm.core.model.count import ProbabilityRatioModel
        model = ProbabilityRatioModel(corpus, window_radius, freq_dist)
    elif model_type == "ppmi":
        from .ldm.core.model.count import PPMIModel
        model = PPMIModel(corpus, window_radius, freq_dist)
    # Predict vector models:
    elif model_type == "skip-gram":
        from .ldm.core.model.predict import SkipGramModel
        model = SkipGramModel(corpus, window_radius, embedding_size)
    elif model_type == "cbow":
        from .ldm.core.model.predict import CbowModel
        model = CbowModel(corpus, window_radius, embedding_size)
    else:
        raise NotImplementedError()
    return model


if __name__ == '__main__':
    main()
