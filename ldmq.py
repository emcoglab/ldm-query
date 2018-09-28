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
from enum import Enum, auto
from os import path
import argparse

from ldm.core.utils.maths import DistanceType

_corpora = [
    "bnc",
    "subtitles",
    "ukwac",
]

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

class Mode(Enum):
    """The operation mode of the program."""

    # Frequency of word in corpus
    Frequency = auto()

    # Rank of word in corpus
    Rank = auto()

    # Vector representation of word in model
    Vector = auto()

    # Compare words in model
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

class LDMQ:

    readme_path = path.join(path.dirname(path.realpath(__file__)), 'README.md')

    def __init__(self, mode: Mode):
        self.mode: Mode = mode

    def run(self):
        raise NotImplementedError()


if __name__ == '__main__':

    # region Set up args

    argparser = argparse.ArgumentParser(
        description="Query corpora and linguistic distributional models. See README.md for more info.")

    argparser_mode = argparser.add_mutually_exclusive_group()

    for ldmq_mode in Mode:
        argparser_mode.add_argument(ldmq_mode.option, action="store_true")

    argparser.add_argument("--corpus",
                           type=str,
                           choices=_corpora,
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
    argparser.add_argument("--word",
                           type=str,
                           required=False,
                           help="The word to look up.")
    argparser.add_argument("--word-pair",
                           type=str,
                           nargs=2,
                           required=False,
                           metavar=('FIRST WORD', 'SECOND WORD'),
                           help="The words to compare.")
    argparser.add_argument("--words-from-file",
                           type=str,
                           required=False,
                           metavar="PATH",
                           help="The word to look up or compare.")
    argparser.add_argument("--word-pairs-from-file",
                           type=str,
                           required=False,
                           metavar="PATH",
                           help="The word pairs to compare.")
    argparser.add_argument("--output-file",
                           type=str,
                           required=False,
                           metavar="PATH",
                           help="Write the output to this file.  Will overwrite existing files.")

    # endregion

    # region Parse args

    args = argparser.parse_args()

    # Get mode

    # Check not more than one mode
    if sum(1 for m in [args.frequency, args.rank, args.vector, args.compare] if m) > 1:
        argparser.error("Please specify only one mode.")
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
    
    # Validate other options

    if mode is Mode.Frequency:


    # endregion

    argparser.print_help()
    print(args)
