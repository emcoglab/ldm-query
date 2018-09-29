LDMQ: query corpora and linguistic distributional models
========================================================

LDMQ is a Python program which lets you query corpora and linguistic distributional models (LDMs).

For now it is controlled using a command line interface, and requires you to install Python and a few modules.


Installation
------------

LDMQ requires Python 3.6+ and the following modules:

-   `numpy`
-   `scipy`
-   `gensim`
-   `pyyaml`
-   `matplotlib`
-   `nltk`
-   `pandas`
-   `scikit-learn`
-   `seaborn`
-   `sklearn`
-   `srt`
-   `statsmodels`

You can download Python from [its website][0], or use a distribution and package manager like [Conda][1].

Once Python is installed, you can install each of the above modules using a package manager like `pip` or `conda`.  
These are usually used from the command line something like this:

    conda install numpy
    conda install scipy
    ...
    
Finally, once all these modules are installed, you can run LDMQ from the command line like:

    python ldmq.py


Configuration
-------------

Before LDMQ can be properly used, it must be configured so it knows where the files containing the corpora and LDMs are 
stored on your computer.

These are set in the file `config.yaml`, which is a text file in [YAML][2] format.  Comments in that file should explain
to set your preferences.
    
    
Usage: overview
---------------

LDMQ can do several things:

-   Count occurrences of a token within a corpus
-   Look up the rank frequency of a token
-   Get the vector representation of a token from a vector-based LDM
-   Compare a pair of tokens in a vector-based LDM using a given distance measure
-   Compare a pair of tokens in an n-gram-based LDM

In general, the model of usage is as follows:

    python ldmq.py \
        <mode> \
            --corpus <corpus-name> \
            --model <model-name> [<embedding size>] \
            --radius <window-radius> \
            --distance <distance-type> \
            --word(-pair) "<first-word>" ("<second-word>")
                       
where some of the arguments are optional depending on context.  (Here, the backslashes \ just "escape" line breaks in the
command line.)  So for example:

    python ldmq.py --frequency --corpus subtitles --word "house"

would look up the frequency of the word "house" in the "BBC subtitles" corpus (giving the answer XXX).

The available `<mode>`s are:

-   `frequency`: Returns the frequency a word in a corpus.
-   `rank`: Returns the rank frequency of a word in a corpus (1 means most frequent).
-   `vector`: Returns the vector representation of a word in a vector-based LDM.
-   `compare`: Compares two words using an LDM.

Each of the available usage modes and options are explained in full, with examples, below.


Usage: options
--------------

Options refer to parts of the command line invocation after the mode.  These will always be a double hyphen, followed 
immediately by the option name, then with extra parameters following it, separated by spaces.  For example, to compare 
the words "cat" and "dog" in the PPMI n-gram model trained on the BNC corpus with window radius 3, we would use the 
command:

    python ldmq.py \
        compare \
            --corpus bnc \
            --model ppmi-ngram \
            --radius 3 \
            --word-pair "cat" "dog"

which would produce the result XXX.

Options can be given in any order after the mode, so we would get same result by running:

    python ldmq.py \
        compare \
            --word-pair "cat" "dog" \
            --model ppmi-ngram \
            --radius 3 \
            --corpus bnc

Some options can be used with several usage modes (such as `--corpus`, which is used with every mode), and some are 
specific (such as `--distance`, which is only used with the `compare` mode when using a vector-based LDM).
    
These are all the options, what they mean, what values they can take, and what modes they are used with.

-   `--corpus <corpus-name>`: The corpus `<corpus-name>` will be used.
    Required in all LDMQ modes.
    The permissible values of `<corpus-name>` are:
    -   `bnc` to use the 100-million-word BNC corpus.
    -   `subtitles` to use the 200-million-word BBC Subtitles corpus.
    -   `ukwac` to use the 2-billion-word UKWAC corpus.
    
-   `--model <model-name> [<embedding-size>]`: The specification of the LDM to be used.
    Required for `vector` and `compare` modes.
    The permissible values of `<model-name>` are:
    -   N-gram models:
        -   `log-ngram`: The log n-gram count model.
        -   `probability-ratio-ngram`: The probability ratio n-gram model.
        -   `ppmi-ngram`: The positive pointwise mutual information (PPMI) n-gram model.
    -   Count vector models:
        -   `log-cooccurrence`: The log co-occurrence count vector model.
        -   `conditional-probability`: The conditional probability vector model.
        -   `probability-ratio`: The probability ratio vector model.
        -   `ppmi`: The PPMI vector model.
    -   Predict vector models:
        -   `skip-gram`: The skip-gram model from word2vec.
        -   `cbow`: The continuous bag of words (CBOW) model from word2vec.
        
    Following the model name, a number is given to specify the embedding size (for predict vector models only).
    
        python ldmq compare --word-pair "cat" "dog" --corpus bnc --distance cosine --model cbow 300 --radius 5
        
    would compare the words "cat" and "dog" using cosine distance using the CBOW model with embedding size 300 and 
    window radius 5, trained on the BNC.  Whereas
    
        python ldmq compare --word-pair "cat" "dog" --corpus bnc --distance cosine --model ppmi --radius 5
        
    would make the same comparison using the PPMI model with window radius 5.
    
-   `--radius <window-radius>`: The context window radius used in the model.
    Required whenever `--model` is used.
    The permissible values of `<window-radius>` are `1`, `3`, `5`, or `10`.
        
-   `--distance <distance-type>`: The distance type for comparing vectors.
    Required for `compare` modes when (and only when) the `--model` is a count vector model or a predict vector model 
    (i.e. not an n-gram model).
    The permissible values of `<distance-type>` are:
    -   `cosine`: Cosine distance.
    -   `correlation`: Correlation distance.
    -   `euclidean`: Euclidean distance.
    
-   `--word "<word>"`: The word to be looked up.
    Only available in `frequency` and `rank` modes.
    The word should be surrounded with double-quote marks.
    Only single-word tokens can be used.  The logic for what counts as a single-word token is quite complex, but (for 
    example) `"single-word"` is a single word word (i.e. hyphens are allowed), and `"two words"` and `"word's"` aren't 
    (i.e. spaces and apostrophes are considered to break a word).
    
-   `--word-pair "<first-word>" "<second-word>"`: Word pair to be compared.
    Only available in the `compare` mode.
    Only single-word tokens can be used (logic same as for `--word`) and words should be surrounded by double quotes.

-   `--words-from-file "<path-to-csv>"`: Words to be used, in a csv.
    Can be used anywhere `--word` can be.  Can also be used in the `compare` mode.
    `<path-to-csv>` should be the absolute path to a csv file, and should be surrounded in double quotes.
    See the "Passing words in CSV format" section for more info on how the CSV should be formatted and what the results
    will be.

-   `--word-pairs-from-file "<path-to-csv>"`: Words to be used, in a csv.
    Can be used anywhere `--word-pair` can be.
    `<path-to-csv>` should be the absolute path to a csv file, and should be surrounded in double quotes.
    See the "Passing words in CSV format" section for more info on how the CSV should be formatted and what the results
    will be.
    
-   `--output-file "<path-to-file>"`: Path to file where results will be written.
    If this optional paramter is given, the results will be written to the specified file rather than displayed in the 
    terminal.
    `<path-to-csv>` should be the absolute path to a csv file, and should be surrounded in double quotes.
    If the file already exists, it will be overwritten.
    

Usage: `frequency` mode
-----------------------

Returns the frequency of a word in the specified corpus.

    python ldmq.py \
        frequency \
            --corpus <corpus-name> \
            --word "<word>"
        
    python ldmq.py \
        frequency \
            --corpus <corpus-name> \
            --words-from-file "<path-to-file>" \
            --output-file "<path-to-file>"

Required options:
-   `--corpus`.
-   Either `--word` or `--words-from-file`.   

Optional options:
-   `--output-file`.

Output:
-   If the `--word` option is used, the output will be the frequency of the word as an integer.
-   If the `--words-from-file` option is used, the output will be csv-formatted, with a two columns:
    -   "Word"
    -   "Frequency in <corpus-name> corpus"
    
Example:



Usage: `rank` mode
------------------

Returns the rank of a word in the specified corpus, by frequency.

    python ldmq.py \
        rank \
            --corpus <corpus-name> \
            --word "<word>"
        
    python ldmq.py \
        rank \
            --corpus <corpus-name> \
            --words-from-file "<path-to-file>" \
            --output-file "<path-to-file>"

Required options:
-   `--corpus`.
-   Either `--word` or `--words-from-file`.   

Optional options:
-   `--output-file`.

Output:
-   If the `--word` option is used, the output will be the rank of the word in the corpus, ordered by frequency, with 1 
    being the most frequent word in the corpus.
-   If the `--words-from-file` option is used, the output will be csv-formatted, with a two columns:
    -   "Word"
    -   "Rank in <corpus-name> corpus"
    
Example:


Usage: `vector` mode
--------------------

Returns the vector representation of a word in the specified vector-based LDM.

    python ldmq.py \
        vector \
            --corpus <corpus-name> \
            --model <model-name> <embedding-size> <window-radius> \
            --word "<word>"
        
    python ldmq.py \
        vector \
            --corpus <corpus-name> \
            --model <model-name> <embedding-size> <window-radius> \
            --words-from-file "<path-to-file>" \
            --output-file "<path-to-file>"

Required options:
-   `--corpus`.
-   `--model`.  Must be a vector-based LDM.
-   Either `--word` or `--words-from-file`.   

Optional options:
-   `--output-file`.

Output:
-   If the `--word` option is used, the output will be its vector representation in the corpus.
-   If the `--words-from-file` option is used, the output will be a matrix in CSV format.  The first column will be 
    "Word" and the remaining columns will be the entries in the vector.
    
Example:


Usage: `compare` mode
---------------------

Returns the comparison score between the specified words in the specified LDM.

    python ldmq.py \
        compare \
            --corpus <corpus-name> \
            --model <model-name> <embedding-size> <window-radius> \
            --distance <distance-type> \
            --word-pair "<first-word>" "<second-word>"
        
    python ldmq.py \
        compare \
            --corpus <corpus-name> \
            --model <model-name> <embedding-size> <window-radius> \
            --distance <distance-type> \
            --word-pairs-from-file "<path-to-file>" \
            --output-file "<path-to-file>"
        
    python ldmq.py \
        compare \
            --corpus <corpus-name> \
            --model <model-name> <embedding-size> <window-radius> \
            --distance <distance-type> \
            --words-from-file "<path-to-file>" \
            --output-file "<path-to-file>"

Required options:
-   `--corpus`.
-   `--model`.
-   `--distance`.  Only if the model is a vector-based LDM.
-   Either `--word-pairs` or `--word-pairs-from-file` or `--words-from-file`.   

Optional options:
-   `--output-file`.

Output:
-   If the `--word-pair` option is used and the model is a vector-based LDM with a `--distance` also provided, the 
    output will be the distance between the word pair in the specified model using the specified distance.  If the model
    is an n-gram model and `--distance` is not provided, the output will be the distributional association between the 
    two words.
-   If the `--word-pairs-from-file` option is used, the output will be a csv with columns:
    -   "First word"
    -   "Second word"
    -   "<distance name> distance" (in case a distance was used) or "Association" (in case it wasn't).
-   If the `--words-from-file` option is used, the output will be a csv-formatted distance matrix with a "First word" 
    column and other columns corresponding to each word provided, with entries being the pairwise comparison of those 
    words in the model.
    
Example:


Passing words in CSV format
---------------------------

-   If using the `--words-from-file` option, the csv should be a single-column file without a header.  I.e., a text file 
    with a single word on each line.
-   If using the `--word-pairs-from-file` option, the csv should be a two-column file without a header.  I.e., a text 
    file with two words on each line, separated by a comma.


--- 
[0]:    https://www.python.org/downloads/
[1]:    https://conda.io/miniconda.html
[2]:    http://yaml.org

TODO: fill in XXXs
TODO: fill in examples