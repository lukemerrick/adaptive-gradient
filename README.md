# adaptive-gradient
Exploration of the efficacy and generalization of solutions found by adaptive gradient methods

This work serves as an initial empirical study of the use of adaptive gradient methods to train Deep Neural Networks (DNNs). Recent state-of-the-art results in computer vision tasks have across the board been achieved using models trained via Stochastic Gradient Descent with Momentum (SGD with Momentum), in spite of the many advantages ascribed to adaptive methods like ADAM. A recent work by Ashia Wilson et. al. demonstrated both theoretically and empirically justification for the use of SGD with Momentum over adaptive methods in the training of DNNs [1]. Sashank Reddi et. al. more recently published a deeper theoretical analysis of a major shortcoming of ADAM and related adaptive methods, and proposed AMSGrad, a derivative of the ADAM algorithm designed to fix these issues [2]. Considering both of these results raises the following question: If ADAM and its ilk are ill-suited to training DNNs, but AMSGrad solves some of the issues of ADAM, are adaptive methods a viable alternative to the simple approach of SGD with Momentum, or will AMSGrad still fall short of SGD with Momentum when it comes to training DNNs? This work is a direct attempt at beginning to answer that question.

The full paper associated with this work was presented at the 2018 Systems and Information Engineering Design Symposium (SIEDS) and is awaiting publication through the IEEE Xplore Digital Library. At that time it will likely show up in search results for ["Luke Merrick"](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&searchWithin=%22First%20Name%22:Luke&searchWithin=%22Last%20Name%22:Merrick).

The notebook "analyzing results.ipynb" allows for the reproduction of the figures and statistical tests found in the paper. To rerun the experiments that generated the data found in the "runs" directory, one simply needs to run the file "sgd_vs_adam.py".


[1] Wilson, Ashia C., et al. "The marginal value of adaptive gradient methods in machine learning." Advances in Neural Information Processing Systems. 2017.

[2] Reddi, Sashank J., Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond." International Conference on Learning Representations. 2018.