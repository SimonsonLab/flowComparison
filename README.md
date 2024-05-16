# flowComparison
Code repository for manuscript "**Comparison of three machine learning algorithms for classification of B-cell neoplasms using clinical flow cytometry data**" (https://doi.org/10.1002/cyto.b.22177)

**Comparison of three machine learning algorithms for classification of B-cell neoplasms using clinical flow cytometry data**

_Wikum Dinalankara, David P. Ng, Luigi Marchionni, Paul D. Simonson_

Abstract

_Multiparameter flow cytometry data is visually inspected by expert personnel as part of standard clinical disease diagnosis practice. This is a demanding and costly process, and recent research has demonstrated that it is possible to utilize artificial intelligence (AI) algorithms to assist in the interpretive process. Here we report our examination of three previously published machine learning methods for classification of flow cytometry data and apply these to a B-cell neoplasm dataset to obtain predicted disease subtypes. Each of the examined methods classifies samples according to specific disease categories using ungated flow cytometry data. We compare and contrast the three algorithms with respect to their architectures, and we report the multiclass classification accuracies and relative required computation times. Despite different architectures, two of the methods, flowCat and EnsembleCNN, had similarly good accuracies with relatively fast computational times. We note a speed advantage for EnsembleCNN, particularly in the case of addition of training data and retraining of the classifier._

Notes for running:

(1) Download and assemble the data (see https://doi.org/10.1002/cyto.a.24159 for details).

(2) For running flowCat, first download and prepare the environment as directed in the repository (https://github.com/xiamaz/flowCat). Edit the shell script in the flowCat/ folder for the paths and execute.

(3) For running EnsembleCNN, run the python scripts and then the shell script in ensembleCNN/src/

(4) For running UMAP-RF, run the python scripts and then the shell script in UMAP_RF/src/



