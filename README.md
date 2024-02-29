<!-- # BadChain
This repo provides the official implementation of ICLR 24 Paper [BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models](https://arxiv.org/abs/2401.12242) -->

<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://zhenxianglance.github.io/" target="_blank" style="text-decoration: none;">Zhen Xiang<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=kTXY8P0AAAAJ&hl=en" target="_blank" style="text-decoration: none;">Fengqing Jiang<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=XL6QafwAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Zidi Xiong<sup>1</sup></a><br>
    <a href="https://sites.google.com/view/rbhaskar" target="_blank" style="text-decoration: none;">Bhaskar Ramasubramanian<sup>3</sup></a>&nbsp;,&nbsp;
  <a href="https://labs.ece.uw.edu/nsl/faculty/radha/" target="_blank" style="text-decoration: none;">Radha Poovendran<sup>2</sup></a>&nbsp;,&nbsp;
  <a href="https://aisecure.github.io/" target="_blank" style="text-decoration: none;">Bo Li<sup>1</sup></a>&nbsp;&nbsp;
    <br/> 
<sup>1</sup>University of Illinois Urbana-Champaign&nbsp;&nbsp;&nbsp;<sup>2</sup>University of Washington&nbsp;&nbsp;&nbsp;<sup>3</sup>Western Washington University&nbsp;&nbsp;&nbsp;<br> 
</p>

<p align='center';>
<b>
<em>ICLR 2024</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2401.12242" target="_blank" style="text-decoration: none;">[arXiv]</a>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://openreview.net/forum?id=c93SBwz1Ma" target="_blank" style="text-decoration: none;">[OpenReview]</a>
</b>
</p>

## Overview
We propose BadChain,the first backdoor attack against LLMs employing COT prompting, which does not require access to the training dataset or model parameters and imposes low computational overhead. BadChain leverages the inherent reasoning capabilities of LLMs by inserting a backdoor reasoning step into the sequence of reasoning steps of the model output, thereby altering the final response when a backdoor trigger exists in the query prompt. In particular, a subset of demonstrations will be manipulated to incorporate a backdoor reasoning step in COT prompting. Consequently, given any query prompt containing the backdoor trigger, the LLM will be misled to output unintended content. Empirically, we show the effectiveness of BadChain for two COT strategies across four LLMs (Llama2, GPT-3.5, PaLM2, and GPT-4) and six complex benchmark tasks encompassing arithmetic, commonsense, and symbolic reasoning.

<p align='center'>
<img src="assets/badchain_fig.png" style="width: 80%;" />
</p>

## Experiment
### Setup
- Make sure to set your api key before running experiment. See the placeholder at `utils.py`

### Running
A example running command is as follow:
```python
python run.py --llm gpt-3.5 --task gsm8k
```
More details can be found at `run.py`
## Citation
If you find our work is useful in your research, please consider citing:
```
@misc{xiang2024badchain,
    title={BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models}, 
    author={Zhen Xiang and Fengqing Jiang and Zidi Xiong and Bhaskar Ramasubramanian and Radha Poovendran and Bo Li},
    year={2024},
    eprint={2401.12242},
    archivePrefix={arXiv},
    primaryClass={cs.CR}
}
```

<br><br>