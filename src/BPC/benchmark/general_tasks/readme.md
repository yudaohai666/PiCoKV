GLUE
NOTE: GLUE benchmark tasks do not provide publicly accessible labels for their test sets, so we default to the validation sets for all sub-tasks.

Paper
Title: GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding

Abstract: https://openreview.net/pdf?id=rJ4km2R5t7

The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. GLUE consists of:

A benchmark of nine sentence- or sentence-pair language understanding tasks built on established existing datasets and selected to cover a diverse range of dataset sizes, text genres, and degrees of difficulty, and
A diagnostic dataset designed to evaluate and analyze model performance with respect to a wide range of linguistic phenomena found in natural language.
Homepage: https://gluebenchmark.com/

Groups, Tags, and Tasks
Groups
None.

Tags
glue: Run all Glue subtasks.
Tasks
cola
mnli
mrpc
qnli
qqp
rte
sst
wnli

SuperGLUE
Paper
Title: SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems Abstract: https://w4ngatang.github.io/static/papers/superglue.pdf

SuperGLUE is a benchmark styled after GLUE with a new set of more difficult language understanding tasks.

Homepage: https://super.gluebenchmark.com/

Groups, Tags, and Tasks
Groups
None.

Tags
super-glue-lm-eval-v1: SuperGLUE eval adapted from LM Eval V1
super-glue-t5-prompt: SuperGLUE prompt and evaluation that matches the T5 paper (if using accelerate, will error if record is included.)
Tasks
Comparison between validation split score on T5x and LM-Eval (T5x models converted to HF)

T5V1.1 Base	SGLUE	BoolQ	CB	Copa	MultiRC	ReCoRD	RTE	WiC	WSC
T5x	69.47	78.47(acc)	83.93(f1) 87.5(acc)	50(acc)	73.81(f1) 33.26(em)	70.09(em) 71.34(f1)	78.7(acc)	63.64(acc)	75(acc)
LM-Eval	71.35	79.36(acc)	83.63(f1) 87.5(acc)	63(acc)	73.45(f1) 33.26(em)	69.85(em) 68.86(f1)	78.34(acc)	65.83(acc)	75.96(acc)
super-glue-lm-eval-v1

boolq
cb
copa
multirc
record
rte
wic
wsc
super-glue-t5-prompt

super_glue-boolq-t5-prompt
super_glue-cb-t5-prompt
super_glue-copa-t5-prompt
super_glue-multirc-t5-prompt
super_glue-record-t5-prompt
super_glue-rte-t5-prompt
super_glue-wic-t5-prompt
super_glue-wsc-t5-prompt