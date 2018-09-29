# Generating Words from Meanings
This is the code for my blog post on Generating Words from Meanings. It uses a character level decoder RNN to convert a word embedding (which represents a meaning) into a word by sampling one character at a time.

# Requirements
`python 3.6`  
`pytorch 0.4.1`

Also needs the following packages:  
* `pytorch-nlp` - for getting word embeddings
* `sacred` - keeping track of configs of training runs and easily writing scripts
* `visdom` - live dynamic loss plots
* `pytorch-utils` - for easily writing training code in pytorch
* `visdom-observer` - interface between `sacred` and `visdom`

Install these using:

`pip install -r requirements.txt`

All the scripts are `sacred` experiments, so they can be run as

`python <script>.py with <config updates>`

# Sampling
If you want to directly sample words from a pretrained network, just go ahead and run
`python sampling.py with word=<your word>`

The `sampling.py` script is used to generate words from a trained model. A pretrained set of weights are present in the `trained_model/` directory, along with the config used to train it.

Run `python sampling.py print_config` to see the different sampling parameters.

# Train your own model
Run `python train.py print_config` to get a list of config options for training your own model.

To train your own model, make sure to have a `visdom` server running in the background at port `8097`. Just run `visdom` in a separate terminal before running the train script.




