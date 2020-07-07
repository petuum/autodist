# Rationale

As ML and deep learning models become more structurally complex, existing distributed ML systems have struggled to provide 
excellent _all-round_ performance on a wide variety of models, since most of them are specialized upon one monolithic system architecture or technology.
For examples, Horovod is built upon MPI and AllReduce, and believed to work well with dense variables like those found in CNNs, but exhibits limitations on 
synchronizing sparse parameter variables like those embedding layers found in many NLP models. In contrast, many parameter server based systems are reported to be better-suited with NLP models than on CNNs. 

Unlike existing systems, AutoDist’s design is motivated by the fact that different ML models (or different components in a complex model) exhibit different runtime characteristics, 
and different learning algorithms demonstrate distinct computational patterns, which demand model and algorithm-aware system or parallelization treatments for distributed execution performance. 
AutoDist achieves such adaptiveness by assembling or interpolating various distributed ML techniques, optimizing distribution strategies against detailed model characteristics and resource specifications.
To this end, AutoDist design focuses on composability, resuability of various ML parallelization techniques, and extensibility to future emerging distributed ML techniques.

AutoDist distributes an incoming model by going through a compilation process, analyzes the computational graph of an incoming model, and generates an appropriate strategy, _just in time_, based on the 
compilation results, and transforms the original single-node computational graph based on the generated strategy to a distributed one, running on distributed clusters. 

For normal users, AutoDist offers up to 8 pre-built distribution strategies, covering basic ones such as parameter server and allreduce, and advanced hybrid 
strategies such as Poseidon and Parallax. The performance is either on par with or better than existing available distributed ML systems, as reported below. [TODO: LINK TO BENCHMARK RESULT]

Beyond pre-built strategies, AutoDist lowers the barrier of creating distribution strategies based on the incoming model of interest. Implementing a custom strategy in AutoDist does not require as much familarity or 
domain knowledge as implementing a distributed ML system from scratch, see details in strategy building tutorial.

