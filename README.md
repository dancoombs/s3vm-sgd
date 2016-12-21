# s3vm-sgd
UIUC ECE598NS "Machine Learning in Silicon" class final project. An online semi-supervised SVM designed to be efficiently implemented in a silicon architecture for a continuous in-situ learning system.

See discussion of project in ece598ns_report.pdf for details

### Project Goals:
1. Explore semi-supervised support vector machines
2. Develop a system that could be built into custom silicon
	a. Online learning
	b. Computationally simple
3. Test system on real world data sets (in progress)

### Code
svm/

1. svm_custom.py : Custom built SVM for learning purposes, trained using quadratic programming
2. s3vm.py : Custom semi-supervised svm trained using SGD (in progress)
3. s3vm_fp.py : Fixed point version of 2. to explore implementation in a custom silicon accelerator
4. qns3vm.py : Open-source qns3vm for comparison

### Notebooks (view ece598ns_report.pdf for details)
notebooks/

1. online_ssl_svm.ipynb : Initial exploration into different online and semi-supervised SVMs.
2. svm_custom_test.ipynb : Test notebook for custom SVM trained using quadratic programming
3. svm_sgd_test.ipynb : Test notebook for custom SVM trained using SGD
4. s3vm_sgd_test.ipynb : Test notebook for custom semi-supervised SVM using SGD
5. s3vm_fp_test.ipynb : Test notebook for fixed point version of s3vm_sgd
6. g50c.ipynb : Test notebook for all methods on a higher dimensional dataset
