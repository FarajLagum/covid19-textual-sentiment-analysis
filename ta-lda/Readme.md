# Topic Modeling using LDA

This repository contains code for performing topic modeling using Latent Dirichlet Allocation (LDA) in Python. The code requires the following packages:

- pandas
- nltk
- gensim
- pyLDAvis
- jupytext

To install the packages, run:

```bash
pip install -r requirements.txt
```



## Usage

To run the code, first make sure that you have installed all the required packages. Then, follow these steps:

1. Clone this repository.
2. Navigate to the cloned directory and run Jupyter Notebook by typing `jupyter notebook` in the command line.
3. Open the `ta-lda.ipynb` notebook and run the cells.

Note that `ta-lda.ipynb` is a Jupyter Notebook file that is compatible with Jupytext. This means that you can edit the notebook in either .ipynb or .py format. To convert the notebook to a .py file, use the following command:

```bash
jupytext --set-formats ipynb,py:percent ta-lda.ipynb
```





## Output

The code generates an HTML file that contains an interactive visualization of the topics using pyLDAvis. You can find the visualization results in the directory `results`:

```bash
cd results
```

