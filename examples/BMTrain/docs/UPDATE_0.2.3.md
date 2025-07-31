# Update Log 0.2.3

**Full Changelog**: https://github.com/OpenBMB/BMTrain/compare/0.2.0...0.2.3


## What's New

### 1. Get rid of torch cpp extension when compiling

Before 0.2.3, the installation of BMTrain requires the torch cpp extension, which is not friendly to some users (it requires CUDA Runtime fits with torch). Now we get rid of the torch cpp extension when compiling BMTrain, which makes the source-code way installation of BMTrain more convenient.
Just run `pip install .` to install BMTrain using source code.

### 2. CICD

In 0.2.3, we bring the Github action CICD to BMTrain. Now we can run the CI/CD pipeline on Github to ensure the quality of the code. CICD will run the test cases and compile the source code into wheel packages. 

### 3. Loss scale management

In 0.2.3, we add the min and max loss scale to the loss scale manager. The loss scale manager can adjust the loss scale dynamically according to the loss scale's min and max value. This feature can help users to avoid the loss scale being too large or too small.


### 3. Others

* Fix `bmt.load(model)` OOM when meets torch >= 1.12
* `AdamOffloadOptimizer` can choose avx flag automatically in runtime
* Now BMTrain is fully compatible with torch 2.0
