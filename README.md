# Generating Words from Embeddings
This is the code for my blog post on [Generating Words from Embeddings](https://rajatvd.github.io/Generating-Words-From-Embeddings/). It uses a character level decoder RNN to convert a word embedding (which represents a meaning) into a word by sampling one character at a time.

Play around with sampling words in this [colab notebook](https://colab.research.google.com/drive/1f_vW0k8YyoyiPIgX7eHP_a-8T3Zepat3).

To get the code working on your local machine, run these commands:

```
conda create -y -n word_generator python=3.6
source activate word_generator
git clone https://github.com/rajatvd/WordGenerator.git
cd WordGenerator
pip install -r requirements.txt
python preprocess_data.py
python sampling.py with word=musical sigma=0.2
```

This works only for Linux systems. For Windows, remove `source`, and just use `activate word_generator` in the second line.


# Requirements
`python 3.6`
`pytorch 1.2.0`

Also needs the following packages:
* `pytorch-nlp` - for getting word embeddings [link](https://github.com/PetrochukM/PyTorch-NLP)
* `sacred` - keeping track of configs of training runs and easily writing scripts [link](https://github.com/IDSIA/sacred)
* `visdom` - live dynamic loss plots [link](https://github.com/facebookresearch/visdom)
* `pytorch-utils` - for easily writing training code in pytorch [link](https://github.com/rajatvd/PytorchUtils)
* `visdom-observer` - interface between `sacred` and `visdom` [link](https://github.com/rajatvd/VisdomObserver)

Install these using:

`pip install -r requirements.txt`

All the scripts are `sacred` experiments, so they can be run as

`python <script>.py with <config updates>`

# Word embeddings

First, get the GloVe vectors and preprocess them by running

`python preprocess_data.py`

This will download the GloVe word vectors and pickle them to be used for training and inference.

# Sampling
If you want to directly sample words from a pretrained network, just go ahead and run

`python sampling.py with word=musical sigma=0.2`

You can change the word and _sigma_ to sample for different embeddings. The sampling script also has other parameters like start characters and beam size.

The `sampling.py` script is used to generate words from a trained model. A pretrained set of weights are present in the `trained_model/` directory, along with the config used to train it.

Run `python sampling.py print_config` to see the different sampling parameters.

Examples of words generated. The embedding of the input word + noise is passed into the GRU model to generate the words.

|Input word| Generated words|
|---|---|
| musical    | melodynamic, melodimentary, songrishment  |
| war  | demutualization, armision|
| intelligence  | technicativeness,   intelimetry  |
| intensity  | miltrality, amphasticity   |
| harmony   | symphthism, ordenity, whistlery, hightonial|
| conceptual |  stemanological, mathedrophobic|
| mathematics   | tempologistics, mathdom    |
| research   | scienting  |
| befuddled   | badmanished, stummied, stumpingly   |
| dogmatic   | doctivistic, ordionic, prescribitious, prefactional, pastological    |



# Train your own model
Run `python train.py print_config` to get a list of config options for training your own model.

To train your own model, make sure to have a `visdom` server running in the background at port `8097`. Just run `visdom` in a separate terminal before running the train script to start the server.

To train the same model I used for generating the words in the post, run this command:

`python train.py with trained_model/config.json`






