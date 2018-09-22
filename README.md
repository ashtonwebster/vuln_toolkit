# Software Vulnerability Detection Toolkit

## Overview

This toolkit contains open-source implementations of transfer learning algorithms and performance metrics for software defect and vulnerability prediction. Included with the distribution of this toolkit are the scripts and datasets necessary to replicate the results of [1].

## Installation

* Install Anaconda for python version 2.7
* Run the following commands:
```
cd /path/to/vuln_toolkit
python setup.py install
```

If you are developing, change the second line to:

```
python setup.py develop
```

Which will incorporate changes to source code immediately.

You should now be able to access the packages vuln\_toolkit, vuln\_toolkit.tl\_algs, and vuln_toolkit.common by simply importing the packages like any others.

* Set the environment variable "VULN\_INPUT" in your .bashrc, .profile, or some other file that is set on load.  For example, in my .profile, I have a line like this:

```
export VULN_INPUT=/path/to/this/repo/vuln_toolkit/datasets/
```

## Replication

TODO

## Tutorial

This tutorial provides a brief overview of the datasets, algorithms and performance metrics included with this toolkit, and examples of their use. For additional information, see the documentation for individual classes and functions, in addition to the papers cited in the bibliography.

### Importing Datasets

The module `common.parse_input` defines functions for importing metrics and token data from the PROMISE and PHP Security datasets. For these functions to work correctly, the `VULN_INPUT` environment variable must be set. See the installation instructions for further details on setting up your environment.

The PHP Security dataset [2] contains software engineering metrics and token data for each file in multiple releases of Moodle, phpMyAdmin and Drupal, along with labels indicating which files contain known vulnerabilities. The portion of the PHP Security dataset included with this toolkit contains labeled metrics and token data for Moodle 2.2.0, phpMyAdmin 3.3.0 and Drupal 6.0.

* The function `parse_metric_projects` parses the PHP Security metrics dataset into an array of dictionaries of feature matrices and label vectors indexed by project.

* The function `parse_token_projects` parses the PHP Security token dataset into an array of dictionaries of TF-IDF scores and label vectors indexed by project. By default, the top 1000 tokens by descending TF-IDF scores are selected.

The PROMISE dataset [3] contains software engineering metrics for each file in multiple releases of 33 object-oriented software projects, along with labels indicating the number of known defects in each file.

* The function `parse_promise_csvs` parses the PROMISE dataset into an array of dictionaries of feature matrices and label vectors indexed by project. The number of defects per file is mapped to a boolean value indicating which files contain known defects.

### Partitioning Datasets

For a given target project, the function `split_train_test` partitions a dataset in the format specified above into training and test sets by randomly sampling a given proportion of data from the target project for the test set. All remaining data, including the portion of the target project not sampled for testing data, forms the training set.

For example, the following code partitions the PHP Security metrics dataset into training and test sets by randomly sampling 80% of the Drupal data. The remaining 20% of Drupal data is used as training data.

```
import common.parse_input

projects = parse_input.parse_metric_projects()
X_train, y_train, train_project, X_test, y_test = \
    parse_input.split_train_test(projects, 'drupal', 0.8)
```

### Standardizing Datasets

The module `common.parse_input` also defines functions for standardizing datasets. In addition to improving the time and space efficiency of some classifiers, standardization improves the performance of transfer learning algorithms that depend on geometric properties of the feature space, such as the Burak and Peters filter.

 Given a partition of a dataset into training and test sets, the function `normalize_per_project` standardizes the features of each project with respect to feature means and standard deviations for that project. For example, the cyclomatic complexity of Moodle files in the PHP Security metrics dataset will be standardized with respect to the mean and standard deviation of that feature in the Moodle project, not the Drupal or phpMyAdmin projects.

### Transfer Learning Algorithms

The `vuln_toolkit.tl_algs` module defines three baselines and four transfer learning algorithms. Each algorithm inherits from the class `Base_Transfer`, is parameterized by a base classifier, and implements the function `train_filter_test`. This function trains the classifier on the test data using a transfer learning algorithm and returns a tuple of class predictions and predicted-class probabilities for the test data.

For example, the following code trains a random forest classifier on the training set from the previous examples using the Burak filter, described below. The `train_filter_test` method returns arrays of vulnerability predictions and confidence values for those predictions. The parameters are described in more detail below.

```
import tl_algs.burak

X_train_norm, X_test_norm = \
    parse_input.normalize_per_project(X_train, X_test, 'drupal', train_project)

burak_alg = burak.Burak(
  X_test_norm, `drupal`, X_train_norm, y_train, train_project,
  sklearn.ensemble.RandomForestClassifier,
  classifier_params={'n_estimators': 100, 'n_jobs': -1},
  cluster_factor=100
)
confidence, predictions = burak_alg.train_filter_test()
```

#### Baselines

The module `tl_algs.tl_baseline` defines three performance baselines, each of which trains a classifier on a subset of the training data without performing any additional filtering, weighting or processing. The performance baselines can be used to evaluate the marginal performance benefits of the transfer learning algorithms [1].

* The `Source_Baseline` class implements the Source Baseline, which trains a classifier using only source data and no target data. For example, if Drupal was sampled for testing data, then the Source Baseline would train the classifier using only Moodle and phpMyAdmin data in the training set.

* The `Target_Baseline` class implements the Target Baseline, which trains a classifier using only target data and no source data. For example, if Drupal was sampled for testing data, then the Target Baseline would train the classifier using only Drupal data in the training set, and no Moodle or phpMyAdmin data.

* The `Hybrid_Baseline` class implements the Hybrid Baseline, which trains a classifier using both target and source data. The Hybrid Baseline is equivalent to training the classifier on the entire training set.

#### Filtering Algorithms

The class `tl_algs.Burak` implements the Burak filter, which uses *k*-nearest neighbors to construct a training set that resembles the test set [4]. For each test instance, the *k* unique nearest training instance by Euclidean distance are retained, and the classifier trained on the filtered training set. The intuition is that the filtered training set more closely resembles the test set, which consists of target data sampled from a single project.

The class `tl_algs.Peters` implements the Peters filter, which constructs a training set based on the most influential test instances [5]. First, the nearest test instance from each training instance is computed. For each test instance, the closest training instance that selected that test instance in the first step is retained, and the classifier trained on the set of all such training instances. The intuition is that training data should drive the selection process, as it provides a rich database of information about defects and vulnerabilities.

Both filtering algorithms compute distance between every training and test instance. For large datasets, this can be computationally intractable. Thus, this toolkit implements the approximation proposed in [5]. First, *k*-means is used to cluster instances in the training and test sets. Second, clusters containing no test instances are rejected. Finally, the Burak and Peters filters are applied to each cluster retained in the previous step.

#### Weighting Algorithms

The class `tl_algs.GravityWeight` implements the Transfer Naive Bayes algorithm described in [6], which trains a classifier on weighted training instances. By analogy with Newton's Law of Universal Gravitation, the Transfer Naive Bayes algorithm assigns each training instance a weight directly proportional to its similarity with the test set, and inversely proportional to the square of the distance between it and the test set, according to measures of similarity and distance defined in [6]. Although this weighting scheme is used to construct a prior distribution for a Naive Bayes classifier in [6], the implementation in this toolkit is parametrized over an arbitrary classifier.

#### Ensemble Algorithms

The class `tl_algs.TrBagg` implements the Transfer Bagging (TrBagg) algorithm described in [7], an ensemble method which uses bagging to construct optimal subsets of weak classifiers trained on bootstrap-sampled subsets of training data. The version of TrBagg implemented in this toolkit trains weak classifiers on bootstrap-sampled subsets of the training data, then computes an optimal subset of weak classifiers that minimizes empirical error on the target training data. Test instances are classified according to the majority vote of classifiers in the ensemble, as in standard bagging. The key difference between TrBagg and standard bagging techniques is that weak classifiers are only added to the ensemble if there is evidence to suggest that their inclusion will increase the ensemble's ability to classify target instances correctly.

### Performance metrics

The `common.metrics` module defines standard metrics for evaluating the performance of classifiers such as precision and recall, in addition to special metrics for evaluating the performance of software defect and vulnerability prediction. The following functions accept an array of prediction targets and predicted-class probabilities as inputs. See the documentation of individual functions and methods for further information on the functions and classes described here.

* The function `vuln_metrics.get_aucec_curve` returns the cost-effectiveness curve for the prediction, which gives the proportion of defects or vulnerabilities a reviewer would expect to discover upon inspecting a given proportion of the target test data [7].

* The function `vuln_metrics.aucec_score` returns the Area under the Cost-Effectiveness Curve (AUCEC) score for the prediction, which provides a general measure of the algorithm's performance on the target test data across a range of inspection ratio thresholds [7].

* The function `vuln_metrics.aucec50_score` is similar to `vuln_metrics.aucec_score`, except that twice the under the cost-effectiveness curve up to an inspection ratio of 50% (AUCEC50) is computed. Since most reviewers would not be interested in inspecting more than 50% of a project's code base, AUCEC50 is typically a more useful measure of an algorithm's performance [7].

* The function `vuln_metrics.cm_for_ir_threshold` returns a confusion matrix approximately satisfying a specified inspection ratio threshold, so that the number of predicted positive instances as a proportion of the target project is approximately equal to the specified threshold. Confusion matrices are instances of the clas `vuln_metrics.My_CM`, and define standard performance metrics for classification tasks.

For example, the following code computes the cost-effectiveness curve and several performance metrics for the predictions made above. The confusion matrix is approximated with an inspection ratio threshold of 20%.


```
import common.vuln_metrics

ir, recall = vuln_metrics.get_aucec_curve(y_test, confidence)
aucec50 = vuln_metrics.aucec50_score(y_test, confidence)

cm = vuln_metrics.cm_for_ir_threshold(confidence, y_test, 0.2)
recall = cm.recall()
fscore = cm.f_score()
```

## References

[1] Ashton Webster. 2016. "A Comparison of Transfer Learning Algorithms in Defect and Vulnerability Prediction."

[2] https://seam.cs.umd.edu/webvuldata/

[3] http://openscience.us/repo/defect/ck/

[4] Burak Turhan, Tim Menzies, Ayse B. Benar and Justin di Stefano. 2009. “On the Relative Value of Cross-Company and Within-Company Data for Defect Prediction. *Empirical Software Engineering 14, 5*, 540–78.

[5] Fayola Peters, Tim Menzies, and Andrian Marcus. 2013. "Better cross company defect prediction."" *IEEE International Working Conference on Mining Software Repositories*, 409-18.

[6] Ying Ma, Guangchun Luo, Xue Zeng, and Aiguo Chen. 2012. "Transfer Learning for Cross-Company Software Defect Prediction." *Information and Software Technology 54, 3*, 238-56.

[7] Toshihiro Kamishima, Hamasaki Masahiro, and Akaho Shotaro. 2009. "TrBagg: A Simple Transfer Learning Method and its Application to Personalization in Collaborative Tagging." *Proceedings - IEEE International Conference on Data Mining, ICDM*, 219-28.

[6] James Walden, Jeff Stuckman, and Ricardo Scandariato. 2014. "Predicting Vulnerable Components: Software Metrics vs. Text Mining." In *2014 IEEE 25th International Symposium on Software Reliability Engineering*, 23-33.
