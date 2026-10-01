# Data policy

The public repository does not include the raw tweet dataset or the downloaded archive.

The experiments use the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). Download it according to the dataset's terms and place the CSV at:

```text
data/original/Tweets.csv
```

The original project recorded dataset provenance in `data/original/provenance.json`. Processed CSVs, encoded arrays, and split artifacts are generated locally and ignored by Git.
