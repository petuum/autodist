# Rationale

As machine learning models become more structurally complex, existing distributed ML systems have struggled to provide 
excellent _all-round_ performance on a wide variety of models, since most of them are specialized upon one monolithic system architecture or technology.
For examples, Horovod is built upon MPI and AllReduce, and believed to work well with dense parameter variables like those found in CNNs, but exhibits limitations on 
synchronizing sparse parameter variables like those embedding layers found in many NLP models. In contrast, many parameter server based systems are reported to be better-suited with NLP models than on CNNs. 


Unlike existing systems, AutoDist’s design is motivated by the fact that different ML models (or different components in a complex model) exhibit different runtime characteristics, and different learning algorithms demonstrate distinct computational patterns, which demand model and algorithm-aware system or parallelization treatments for distributed execution performance, illustrated below.

<p align="center">
  <image src="image/motivation.png" width=500/>
</p>


AutoDist achieves such adaptiveness by composing various, atomic distributed ML techniques together as a _distribution strategy_, with respect to detailed model characteristics and resource specifications -- AutoDist distributes an incoming model by going through a compilation process, analyzes the computational graph of an incoming model, and generates an appropriate strategy, _just in time_, based on the compilation results, and transforms the original single-node computational graph based on the generated strategy to a distributed one, running on distributed clusters. AutoDist design focuses on composability, resuability of individual ML parallelization building block, and extensibility to future emerging distributed ML techniques.

The figure below contrasts AutoDist (lower) with most existing distributed ML systems (upper). 
<p align="center">
  <img src="images/others.png" width="450"/>
</p>
<p align="center">
  <img src="images/autodist-arch.png" width="450"/>
</p>





