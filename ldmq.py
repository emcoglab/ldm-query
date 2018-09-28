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
import sys
from enum import Enum, auto
from os import path
import argparse

import yaml
from pandas import DataFrame, read_csv

from ldm.core.corpus.corpus import CorpusMetadata
from ldm.core.model.base import DistributionalSemanticModel
from ldm.core.utils.maths import DistanceType

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
    SingleWord = auto()
    SingleWordList = auto()
    WordPair = auto()
    WordPairList = auto()

class LDMQ:

    def __init__(self,
                 mode: Mode,
                 word_mode: WordMode,
                 corpus: str,
                 model: DistributionalSemanticModel = None,
                 distance: DistanceType = None,
                 word_or_words_or_filename = None,
                 from_file: bool = False,
                 output_file_path = None,
                 ):

        # Config
        with open(_config_path, mode="r", encoding="utf-8") as config_file:
            config = yaml.load(config_file)

        self.mode: Mode = mode
        self.word_mode: WordMode

        self.corpus: CorpusMetadata = CorpusMetadata(
            name=_corpora[corpus],
            path=config["corpora"][corpus]["path"],
            freq_dist_path=config["corpora"][corpus]["index"])

        self.model = model

        self.distance: DistanceType = distance

        self.in_file_path: str = word_or_words_or_filename if from_file else None

        self.word_or_word_pair = word_or_words_or_filename if not from_file else None

        self.out_file_path: str = output_file_path


    def run(self):

        if word_mode is WordMode.SingleWord:
            word = self.word_or_word_pair
            if mode is Mode.Frequency:
                ...
            elif mode is Mode.Rank:
                ...
            elif mode is Mode.Vector:
                ...
            else:
                raise NotImplementedError()
        elif word_mode is WordMode.WordPair:
            word_1 = self.word_or_word_pair[0]
            word_2 = self.word_or_word_pair[1]
            if mode is Mode.Frequency:
                ...
            elif mode is Mode.Rank:
                ...
            elif mode is Mode.Vector:
                ...
            else:
                raise NotImplementedError()
        elif word_mode is WordMode.SingleWordList:
            with open(self.in_file_path, mode="r") as in_file:
                word_list = [line.strip() for line in in_file]
            if mode is Mode.Frequency:
                ...
            elif mode is Mode.Rank:
                ...
            elif mode is Mode.Vector:
                ...
            else:
                raise NotImplementedError()
        elif word_mode is WordMode.WordPairList:
            with open(self.in_file_path, mode="r") as in_file:
                word_list_df: DataFrame = read_csv(in_file, header=None)
                word_pair_list = [
                    (row[0], row[1])
                    for row in word_list_df.iterrows()
                ]
            if mode is Mode.Frequency:
                ...
            elif mode is Mode.Rank:
                ...
            elif mode is Mode.Vector:
                ...
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


if __name__ == '__main__':

    # region Set up args

    argparser = argparse.ArgumentParser(
        description="Query corpora and linguistic distributional models. See README.md for more info.")

    argparser_mode = argparser.add_mutually_exclusive_group()
    for ldmq_mode in Mode:
        argparser_mode.add_argument(ldmq_mode.option, action="store_true")

    argparser_wordmode = argparser.add_mutually_exclusive_group()
    argparser_wordmode.add_argument("--word",
                           type=str,
                           required=False,
                           help="The word to look up.")
    argparser_wordmode.add_argument("--word-pair",
                           type=str,
                           nargs=2,
                           required=False,
                           metavar=('FIRST WORD', 'SECOND WORD'),
                           help="The words to compare.")
    argparser_wordmode.add_argument("--words-from-file",
                           type=str,
                           required=False,
                           metavar="PATH",
                           help="The word to look up or compare.")
    argparser_wordmode.add_argument("--word-pairs-from-file",
                           type=str,
                           required=False,
                           metavar="PATH",
                           help="The word pairs to compare.")

    argparser.add_argument("--corpus",
                           type=str,
                           choices=_corpora.keys(),
                           required=True,
                           help="The name of the corpus.")
    argparser.add_argument("--model",
                           type=str,
                           choices=_models,
                           nargs="+",
                           required=False,
                           metavar=("MODEL", "[EMBEDDING] RADIUS"),
                           help="The model to use.")
    argparser.add_argument("--distance",
                           type=str,
                           choices=[dt.name for dt in DistanceType],
                           required=False,
                           help="The distace type to use.")


    argparser.add_argument("--output-file",
                           type=str,
                           required=False,
                           metavar="PATH",
                           help="Write the output to this file.  Will overwrite existing files.")

    # endregion

    # region Parse args

    args = argparser.parse_args()

    # Get mode
    if args.frequency:
        mode = Mode.Frequency
    elif args.rank:
        mode = Mode.Rank
    elif args.vector:
        mode = Mode.Vector
    elif args.compare:
        mode = Mode.Compare
    else:
        raise NotImplementedError()

    # Get word_mode
    # and words or path
    word_mode: WordMode.SingleWord
    if args["word"] is not None:
        word_mode = WordMode.SingleWord
        words_or_path = args["word"]
    elif args["word-pair"] is not None:
        word_mode = WordMode.WordPair
        words_or_path = args["word-pair"]
    elif args["words-from-file"] is not None:
        word_mode = WordMode.SingleWordList
        words_or_path = args["words-from-file"]
    elif args["word-pairs-from-file"] is not None:
        word_mode = WordMode.WordPairList
        words_or_path = args["word-pairs-from-file"]
    else:
        raise NotImplementedError()

    distance: DistanceType
    if args.distance is None:
        distance = None
    elif args.distance.lower() is "cosine":
        distance = DistanceType.cosine
    elif args.distance.lower() is "correlation":
        distance = DistanceType.correlation
    elif args.distance.lower() is "euclidean":
        distance = DistanceType.Euclidean
    else:
        raise NotImplementedError()


    # Get corpus
    corpus = args.corpus.lower()

    # Get model params
    if args.model is None:
        model_type = None
        embedding_size = None
        radius = None
    elif len(args.model) == 0:
        model_type = None
        embedding_size = None
        radius = None
        argparser.error("Please specify model.")
    elif len(args.model) == 1:
        model_type = None
        embedding_size = None
        radius = None
        argparser.error("Please specify window radius")
    elif len(args.model) == 2:
        model_type = args.model[0]
        radius = int(args.model[1])
        embedding_size = None
    elif len(args.model) == 3:
        model_type = args.model[0]
        embedding_size = int(args.model[1])
        radius = int(args.model[3])
    else:
        raise NotImplementedError()

    # Validate options

    if mode is Mode.Frequency:
        if word_mode is WordMode.WordPair or word_mode is WordMode.WordPairList:
            argparser.error("Only use --word or --words-from-file in --frequency mode.")
        if model_type is not None:
            argparser.error("Not valid to use mode in --frequency mode.")

    elif mode is Mode.Rank:
        if word_mode is WordMode.WordPair or word_mode is WordMode.WordPairList:
            argparser.error("Only use --word or --words-from-file in --rank mode.")
    elif mode is Mode.Vector:
        if word_mode is WordMode.WordPair or word_mode is WordMode.WordPairList:
            argparser.error("Only use --word or --words-from-file in --vector mode.")
    elif mode is Mode.Compare:
        if word_mode is WordMode.SingleWord or word_mode is WordMode.SingleWordList:
            argparser.error("Only use --word-pair or --word-pairs-from-file in --frequency mode.")
    else:
        raise NotImplementedError()

    # Build and validate model
    if model_type is None:
        model = None
    else:
        # Validate embedding size
        if (args.model[0] is "log-ngram"
                or args.model[0] is "probability-ratio-ngram"
                or args.model[0] is "ppmi-ngram"
                or args.model[0] is "log-cooccurrence"
                or args.model[0] is "conditional-probability"
                or args.model[0] is "probability-ratio"
                or args.model[0] is "ppmi"):
            if embedding_size is not None:
                argparser.error("Embedding size specified but not valid with N-gram or count models")
                
        if
            ...
        elif args.model[0] is "skip-gram":
            ...
        elif args.model[0] is "cbow":
            ...
        else:
            raise NotImplementedError()

    # endregion

    # region Start application

    app = LDMQ(
        mode=mode,
        word_mode=word_mode,
        corpus=corpus,
        )

    app.run()

    # endregion

    sys.exit(0)
