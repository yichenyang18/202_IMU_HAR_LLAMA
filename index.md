# 202 Project: IMU-based LLMs HAR Performance on Edge Device

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
    1. [Motivation and Objective](#motivation-and-objective)
    2. [Prior Work](#prior-work)
    3. [Novelty of This Work](#novelty-of-this-work)
3. [Technical Approach](#technical-approach)
4. [Evaluation and Results](#evaluation-and-results)
5. [Discussions and Conclusions](#discussions-and-conclusions)
6. [References](#references)



## Abstract
In this project, we will explore and try to augment LLMs' reasoning and cognition in relation to the physical world on edge devices (smartphones), 
evaluated by the performance on Human Activity Recognition (HAR) classification task based on the 
data from inertail measurement units (IMU). We will implement techniques including paramter fine-tuning, 
model prompt configuration, data enrichment, etc and meausure the accuracy, latency, and memory usage.


## Introduction

### Motivation and Objective
In recent years, Large Language Models (LLMs) have demonstrated impressive capabilities
in understanding and generating human language. However, their potential for reasoning
and cognition in relation to the physical world, particularly when dealing with sensor
data from Internet of Things (IoT) devices, remains an area of exploration. IoT devices, 
especially in applications like Human Activity Recognition (HAR), generate complex sensor 
data that LLMs could potentially process for higher-level reasoning and pattern recognition. 
Our project seeks to explore how LLMs can leverage this sensor data to improve understanding 
and classification of physical activities, while addressing key challenges such as 
computational resource limitations, memory usage, and latency on edge devices like smartphones.

Our goal is to assess the performance of LLMs for IMU-based HAR classification tasks on 
smartphones, optimizing them for resource-constrained environments. We will explore techniques 
such as parameter fine-tuning, prompt configuration, data augmentation, etc. to balance 
accuracy, latency, and memory usage. By analyzing trade-offs between model complexity and 
real-world performance, we aim to identify strategies that enhance classification accuracy 
while ensuring efficient deployment on smartphones.

### Prior Work
In research paper "IoT LLM: Enhancing Real World IoT Task Reasoning With Large Language Models" that is published in 2024, 
the authors explore how LLMs can augment reasoning in analyzing IoT device data and performing 
tasks such as industrial anomaly detection, indoor localization, and human activity recognition. 
However, only accuracy is considered in this paper, and other factors such as latency is not 
studied.

### Novelty of This Work
Since the majority of sensor data is collected by edge devices, including smartphone, 
smart home devices, industrial IoT devices, it is convenient and intuitive to use LLMs 
that are deployed on these devices to analyze data and perform related tasks. In addition, 
since LLMs are generally very large, performance-related factors, such as latency and 
computational resource consumption, is also what we consider. Thus, inspired by prior 
works, we want to explore how we can explore and augment LLMs' reasoning from and towards 
real world while addressing the challenges associated with deploying them on edge devices.


## Technical Approach

### Model fine-tuning
[NEED TO COMPLETE]

### Performance test on smartphone
The objective of this part is to explore techniques for augmenting LLM's capability to perform HAR tasks on edge device 
(Android) and analyze the influence of these techniques with multiple factors
(total duration, prompt evaluation rate, response evaluation rate, memory usage, etc.). In consideration of 
the processing latency, we choose to evaluate LLMs' performance on HAR binary classification.

#### Hardware description
In this project, we select One plus 9 pro, an Android device published in 2021.
- CPU: Qualcomm Snapdragon 888 (8 cores)
- GPU: Adreno 660
- RAM: 12GB
- Storage: 256GB

#### Model deployment
In order to deploy our model on Android smartphone, we use Termux platform, which is a power 
terminal emulator and Linux environment app for Android and provides a wide range of tools typically 
available on full-fledged Linux system. It can be downloaded from https://github.com/termux/termux-app/releases.

Once Termux is successfully set up, we use it to deploy Ollama first, which is platform designed to 
make it easier for developers to run LLMs locally. It can be downloaded from https://github.com/ollama/ollama.

After Ollama is set up, we could use `ssh`/`scp` then to send local models from computer to the 
phone and remotely manipulate Termux. For the details in model deployments, please look at [this tutorial](./tutorial.md).

For testing, we have written a Shell script that can automatically feed input files to the LLM.

<img src="./src/ollama_android.jpg" title="Model testing on Android system" width="500" />

(Model testing on Android system)

#### Baseline performance
For the baseline test of our model, we generate different test input files based on the raw 
data, with different data points (1 - 6 data points). The raw data is collected at a frequency of 50 Hz, meaning that 
it has 50 data points per second. In consideration of both processing lantency and key features contained 
in the data, we downsampled the data to 2.5Hz, meaning that that time interval between any two data points 
is 0.4s.

Below is a sample input:
```
The body acceleration on X-axis is [-0.0889], on Y-axis is [0.3492], on Z-axis is [0.292];
the body gyroscope data on X-axis is [0.0585], on Y-axis is [0.0347], on Z-axis is [-0.1154]. 
Please analyze what activity the person is doing. (options: WALKING, SITTING). Return your analysis 
and add your answer as "ANSWER: (ONLY ONE OF THE OPTIONS THAT YOU CHOOSE)" at the end (no need for any coding).

```

#### Model prompt configuration with domain knowledge
Based on the baseline, we aim to augment the model using prompt configuration. As LLMs possess strong role-playing capabilities, 
custimizing LLMs with prompt configuration may invoke the interval knowledge of them, leading to an 
increase of accuracy. In the `.makefile` file that we use to customize LLM, we have defined the specific role that we 
want the LLM to act like, included IMU-related domain knowledge, and finally asked the model to perform classification by 2 step: 
first freely analyze the given input data, and then make the conclusion based on the analysis. Below is the template for 
prompt configuration.

```
SYSTEM """
{Role definition}


{Expert}


{Examples}


{Question definition}


{Response Format}
"""
```

#### Data augmentation 1
Since LLMs are not specifically designed to handle numerical data study, it is not surprising that raw data cannot achieve a satisfying result.
Because IMU data alone can be insufficient for perform tasks like HAR through IMU data, we thus tried to provide a comprehensive text analysis towards the distribution of the data 
also the meaning for the units, along with our raw data. Below is the template we use as the input prompt.

```
Template used for input prompt generation:

Triaxial acceleration data: {data with unit}
X-axis mean:              X-axis variance:
Y-axis mean:              Y-axis variance:
Z-axis mean:              Z-axis variance:


Triaxial gyroscope data: {data with unit}
X-axis mean:              X-axis variance:
Y-axis mean:              Y-axis variance:
Z-axis mean:              Z-axis variance:

(Extra knowledge) The data is collected from an IMU with a frequency of {...}Hz. The unit for acceleration is {...}, 
meaning {...}. The unit for gyroscope is {...}, meaning {...}. 

```

#### Data augmentation 2
Another way for data augmentation is to provide frequency-domain features. Time series data can be confusing for 
LLMs to extract important features. On the other hand, frequency analysis methods are used 
to reveal patterns in signals that are not easily visible in the time domain. They help 
in identifying the underlying frequency structure, filtering noise, and extracting features 
for tasks like classification. Below is the signal processing workflow:
- Median filter
- 3rd Butterworth low-pass filter with a cutoff frequency of 20Hz
- Fast Fourier Transform (FFT)
- Extract key features: e.g. total spectrum energy

![](./src/filtered.png)
![](./src/fft.png)

## Evaluation and Results

### Model fine-tuning
[NEED TO COMPLETE]

### Performance test on smartphone
#### Baseline
![](./src/base1.png)
![](./src/base_td.png)
![](./src/base_ld.png)
![](./src/base_pec.png)
![](./src/base_ped.png)
![](./src/base_per.png)
![](./src/base_ec.png)
![](./src/base_ed.png)
![](./src/base_er.png)

The total test is performed on a relatively small dataset, around 120 input prompts in total. 
The accuracy varies from aroud 0.4 - 0.7, indicating a not good result as we have expected to see. 
In terms of lantency, we can see that even though prompt evaluation duration is doubled
as the number of data points increases from 1 to 6, but the total duration/lantency is not largely 
effected, as the (response) evaluation duration takes a larger proportion in total duration in this 
case.

#### Model prompt configuration with domain knowledge
![](./src/pr_acc.png)
![](./src/pr_td.png)
![](./src/pr_pec.png)
![](./src/pr_ped.png)
![](./src/pr_per.png)
![](./src/pr_ec.png)
![](./src/pr_ed.png)
![](./src/pr_er.png)

After custimizing the model with prompt and domain knowledge, we can see a general increase in accuracy, 
meaning that by providing LLM with domain-specific knowledge and question analysis logic, we can improve its 
reasoning towards analyzing real-world situation.

In terms of latency, we discover a surge in prompt evaluation count and duration even though we have kept the input data 
unchanged, indicating that the custimization file makes the LLM process more tokens when dealing with the same inputs. 
However, it appears that the total duration is not influenced, even has a potential to decrease, due to the smaller response 
evaluation count and duration. We believe that this is because the prompt configuration makes it easier for the model to 
capture the key features in the inputs, leading to less analysis outputs but a better accuracy compared to the baseline.

#### Data augmentation 1
![](./src/da1_acc.png)
![](./src/da1_td.png)
![](./src/da1_ped.png)

Although we have only tested date enriched inputs in a small scale of tests, 
the results already appear to be undesirable. While the accuracy does not show an 
increase compared to previous results, the total duration (mainly prompt evaluation duration) 
surges. 

We believe that the reason behind this phenomenon is that ss the input complexity increases, the model may struggle to determine which parts of the 
input are critical for the task, resulting in little or no improvement in accuracy.

#### Memory usage
<img src="./src/memory_usage.jpg" width="500" />

System Memory Status:
- Total system memory: 11.0 GiB
- Free system memory: 4.7 GiB
- Free swap memory: 3.1 GiB

Model Memory Status:
- Full memory required: 3.3 GiB
- Partial memory required: 0 B 
- Key-value memory required: 896.0 MiB
- Weights total memory: 2.4 GiB
  - Repeating weights: 2.1 GiB
  - Non-repeating weights: 308.2 MiB
- Graph memory requirements:
  - Full graph: 424.0 MiB
  - Partial graph: 570.7 MiB

## Discussions and Conclusions
[NEED TO COMPLETE]


## References
1. An, Tuo, et al. IoT-LLM: Enhancing Real-World IoT Task Reasoning with 
Large Language Models. arXiv:2410.02429, arXiv, 4 Oct. 2024. arXiv.org, 
https://doi.org/10.48550/arXiv.2410.02429.
2. Ji, Sijie, et al. HARGPT: Are LLMs Zero-Shot Human Activity Recognizers? 
arXiv:2403.02727, arXiv, 5 Mar. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2403.02727.

