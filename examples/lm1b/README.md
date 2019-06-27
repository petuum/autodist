# LM-1B 
LM-1B implements the LSTM language model described in [LM](https://arxiv.org/abs/1602.02410). 
The original code comes from [here](https://github.com/rafaljozefowicz/lm), which supports 
synchronous training with multiple GPUs. 

## Dataset
* [1B Word Benchmark Dataset](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark)
* [Vocabulary](https://github.com/rafaljozefowicz/lm/blob/master/1b_word_vocab.txt)

## To Run
You can run lm1b model with data in `<data_dir>` by executing: 
```shell
$ python lm1b_train.py --datadir <data_dir>
```

Also, we have a few more options you can choose for distributed running.

| Parameter Name       |  Default            	| Description |
| :------------------- |:-----------------------| :-----------|
| --logdir			   | /tmp/lm1b				| Logging directory |
| --datadir			   | None					| Data directory |
| --hpconfig		   | ""						| Overrides default hyper-parameters |
| --eval_steps		   | 70						| Number of evaluation steps |
| --max_steps 		   | 1000000    		    | Number of iterations to run for each workers |
| --log_frequency 	   | 100  		    		| How many steps between two runop log |
| --ckpt_dir           | None					| Directory to save checkpoints |
| --save_ckpt_steps    | 0						| Number of steps between two consecutive checkpoints |
| --save_n_ckpts_per_epoch | -1					| Number of checkpoints to save per each epoch |
