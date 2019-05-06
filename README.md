## SCHNEL
**S**calable **C**lustering of **H**ierarchical Stochstic **N**eighbour **E**mbedding hierarchies using **L**ouvain community detection.

Clustering of high dimensional data using manifold learning and subsampling. Clustering is performed on a small representative subset of the data and translated back to the full dataset.

See the notebooks in the Python Parser folder for explanation on the HSNE datastructure, how to parse the HSNE hierarchy into a python object and a clustering [example](https://github.com/paulderaadt/HSNE-clustering/blob/master/PythonParser/Louvain_clustering_example.ipynb).



hsne_cmd, Instead of launching an interactive HSNE, simply export the full HSNE hierarchy to name.hsne in cwd.

Input is a csv file without headers or class labels. Hsne_cmd input first needs to be converted to a binary file.


use:

	.data_conversion/csv_2_bin ../data/MNIST_1000.csv MNIST_1000.bin
	#rows:	1000
		columns:	784
		#rows:	1000

	# Create HSNE hierarchy with 2 scales and save it to mnis_aoi.hsne
	./hsne_cmd ../../../data/MNIST_1000.bin 1000 784 -s 2 -a mnis_aoi
	
	Saving scale:	0
	size	
	... transition matrix ...	
	Saving scale:	1
		size:	238
		... transition matrix ...
		... landmarks to original data ...
		... landmarks to previous scale ...
		... landmark weights ...
		... previous scale to current scale landmarks ...
		... area of influence ...
	Saving scale:	2
		size:	29
		... transition matrix ...
		... landmarks to original data ...
		... landmarks to previous scale ...
		... landmark weights ...
		... previous scale to current scale landmarks ...
		... area of influence ...
	#Read it using the parser:
	python PythonParser/HSNE_parser.py ./PythonParser/sample_data/mnis_aoi.hsne

	Number of scales 3
		Start reading first scale of size 1000
		Done reading first scale..

	Next scale: 1
	Scale size: 236
		Reading transmatrix..
		Reading landmarks of scale to original data..
		Reading landmarks to previous scale..
		Reading landmark weights..
		Reading previous scale to current scale..
		Reading area of influence..

	Next scale: 2
	Scale size: 29
		Reading transmatrix..
		Reading landmarks of scale to original data..
		Reading landmarks to previous scale..
		Reading landmark weights..
		Reading previous scale to current scale..
		Reading area of influence..
	Total time spent parsing hierarchy and building objects: 0.055318
	
HSNE-clustering uses:

Pezzotti N., Hpllt T., Lelieveldt B., Eisemann E., Vilanova A.. _Hierarchical Stochastic Neighbor Embedding Computer_ Graphics Forum. 2016;35:21–30.
https://github.com/Nicola17/High-Dimensional-Inspector

Traag Vincent, Waltman Ludo, Eck Nees Jan. _From Louvain to Leiden: guaranteeing well-connected communities_ arXiv:1810.08473 [physics]. 2018. arXiv: 1810.08473.

https://github.com/vtraag/louvain-igraph
