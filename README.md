# Understanding Tree-based models with Poker Hands

I wrote a [project page](https://ethannguyen2k.github.io/projects/poker/poker.html) about it on my portfolio.

A tree-models repo exploring its logic and mechanism using a dataset of poker hands.

The project involved three pre-built models and one custom-built model:

- Vanilla Decision Tree
- Vanilla Random Forest
- Stacked Ensemble Models (2 Random Forest Classifiers with different parameters, 1 Decision Tree Classifier, and LogisticRegression as the final estimator)
- Custom Decision Tree Model, built from scratch (you can find this at from_scratch notebook)

When I first approached this project, I had a simple goal: understand how decision trees work by building one from scratch. The theory seemed straightforward - split the data based on the most informative feature, repeat until you reach pure leaves. However, the implementation revealed much more complicated the more you have to code for an actual tree-based models.

## The Core Components
1. **Node Class**: Represents each decision point
2. **Information Gain Calculation**: Measures split quality
3. **Tree Building Logic**: Recursive splitting process
4. **Prediction Mechanism**: Traversing the tree for classification

## Attempts
### Before Feature Engineering
- Maximum tree depth: 5 (hard limit)
- Resulting leaf nodes: 32 (result of said hard limit)
- Information gain range: 0.0005 - 0.0103 (with one outlier at 0.0317)

The poor performance was foreseen. The individual card features weren't informative because poker hands are defined by card combinations, not individual cards. A single card's suit or rank tells us very little about the final hand classification.

### After Feature Engineering
- Maximum tree depth needed: Reduced to 3-4
- Leaf nodes: Only 10 needed
- Information gain: Jumped to 0.4304 at root split (41x improvement)

The enhanced feature set included:
- Card frequency counts (tracking pairs, three-of-a-kind, etc.)
- Suit counts (for flush detection)
- Sequential card checks (for straights)
- Maximum card count features (distinguishing between similar pairs or three or four)

## Key Learnings
1. **Feature Engineering Impact**: The right features can dramatically simplify the tree structure while improving accuracy
2. **Information Gain Insights**: Higher gains don't always mean better splits - context matters
3. **Tree Growth Control**: Balancing depth vs. accuracy is crucial for model generalization


References: [Poker Hand - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/158/poker+hand)