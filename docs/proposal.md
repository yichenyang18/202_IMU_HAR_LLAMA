# Project Proposal

## 1. Motivation & Objective

In recent years, Large Language Models (LLMs) have demonstrated impressive capabilities in understanding and generating human language. However, their potential for reasoning and cognition in relation to the physical world, particularly when dealing with sensor data from Internet of Things (IoT) devices, remains an area of exploration. IoT devices, especially in applications like Human Activity Recognition (HAR), generate complex sensor data that LLMs could potentially process for higher-level reasoning and pattern recognition. Our project seeks to explore how LLMs can leverage this sensor data to improve understanding and classification of physical activities, while addressing key challenges such as computational resource limitations, memory usage, and latency on edge devices like smartphones.

Our goal is to assess the performance of LLMs for IMU-based HAR classification tasks on smartphones, optimizing them for resource-constrained environments. We will explore techniques such as parameter fine-tuning, prompt configuration, data augmentation, etc. to balance accuracy, latency, and memory usage. By analyzing trade-offs between model complexity and real-world performance, we aim to identify strategies that enhance classification accuracy while ensuring efficient deployment on smartphones.

## 2. State of the Art & Its Limitations

IMU-based Human Activity Recognition (HAR) on smartphones uses data from Inertial Measurement Units (IMUs) like accelerometers and gyroscopes to classify human activities such as walking, running, and sitting. Current practices involve collecting motion data as time-series signals, preprocessing to remove noise, and segmenting data into fixed-length windows. Traditional methods rely on manual feature extraction, while deep learning models like CNNs and RNNs automatically learn features for improved accuracy. Despite advancements, challenges remain, including generalization across users and environments, variability in smartphone placement and orientation, and energy constraints for real-time processing. Additionally, the reliance on labeled data limits scalability, and recognizing complex or subtle activities remains difficult. Addressing these limitations requires innovations in personalized models, multimodal sensor integration, and efficient algorithms for edge devices.

## 3. Novelty & Rationale

Since the majority of sensor data is collected by edge devices, including smartphone, smart home devices, industrial IoT devices, it is convenient and intuitive to use LLMs that are deployed on these devices to analyze data and perform related tasks. In addition, since LLMs are generally very large, performance-related factors, such as latency and computational resource consumption, is also what we consider. Thus, inspired by prior works, we want to explore how we can explore and augment LLMs' reasoning from and towards real world while addressing the challenges associated with deploying them on edge devices.

## 4. Potential Impact

Technical Impact:

1. Improved Accuracy and Robustness:

    Fine-tuning models would lead to higher recognition accuracy across diverse populations, devices, and environments.
    Robustness to noise, device placement variability, and environmental factors would enable more reliable real-world deployment.

2. Real-Time Efficiency:

    Optimized models would be more computationally efficient, enabling real-time HAR on resource-constrained devices like smartphones and wearables without draining battery life.

3. Advanced Personalization:

    Fine-tuned models could adapt to individual users' unique movement patterns, enhancing the usability of HAR in healthcare, fitness, and lifestyle monitoring.

4. Integration with Multimodal Systems:

    With better performance, IMU-based HAR could complement data from other sensors (e.g., GPS, cameras) for applications requiring multimodal fusion, such as augmented reality or autonomous systems.

Broad Impact:

1. Healthcare Advancements:

    Reliable HAR can improve remote patient monitoring, fall detection for the elderly, and chronic disease management, enhancing preventive care and reducing healthcare costs.

2. Fitness and Wellness:

    Precision in activity tracking would drive innovations in fitness applications, enabling users to receive tailored workout recommendations and progress monitoring.

3. Workplace Safety:

    HAR could monitor workers' movements in hazardous industries, reducing workplace injuries by detecting unsafe postures or fatigue.

4. Accessibility and Assistive Technologies:

    For individuals with disabilities, HAR could power assistive devices that adapt to users' needs by understanding their activity context in real time.

5. Behavioral Insights:

    Reliable activity recognition would enhance human behavior analysis, benefiting fields like psychology, marketing, and human-computer interaction.

## 5. Challenges

1. Data Collection and Labeling:

    Collecting diverse, high-quality IMU datasets with precise labels is labor-intensive and expensive.
    Variability in user behavior, device placement, and movement patterns adds complexity to data collection.

2. Generalization Across Users and Environments:

    HAR systems often fail to generalize across different populations, environmental conditions, and devices due to biases in training data.

3. Computational Constraints:

    Running complex machine learning models in real-time on smartphones requires balancing computational efficiency and energy consumption.

4. Variability in Device Placement:

   Smartphone location (e.g., pocket, hand, bag) and orientation can significantly affect data quality, making it difficult to maintain consistency.

5. Complex Activity Recognition:

   Recognizing subtle, overlapping, or multi-task activities (e.g., talking while walking) remains a significant challenge for most models.

6. Privacy and Security:

   Continuous monitoring of user activities raises concerns about data privacy and the potential misuse of sensitive information.
   Improper data handling can lead to security breaches, exposing users' movement patterns and behavior.

7. Misinterpretation of Data:

   Erroneous activity predictions could lead to incorrect conclusions or actions, especially in sensitive areas like medical diagnostics or law enforcement.


## 6. Requirements for Success

1. Data Collection and Signal Processing

2. Machine Learning and Deep Learning

3. Mobile and Embedded System Development

4. Evaluation and Validation

5. Domain Knowledge on HAR

## 7. Metrics of Success

1. Accuracy performance of fine-tuned models

2. Token Used

3. Memory Used

4. Latency

## 8. Execution Plan

1. Accuracy Performance

  We will test the accuracy performance of models different from the number of epoch, dataset size, the way of data presentation, whether domain knowledge is provided, and the choice of base model.


## 9. Related Work

### 9.a. Papers

List the key papers that you have identified relating to your project idea, and describe how they related to your project. Provide references (with full citation in the References section below).

### 9.b. Datasets

Link provided in Data Folder

### 9.c. Software

Jupyternotebook, Ollama

## 10. References

List references correspondign to citations in your text above. For papers please include full citation and URL. For datasets and software include name and URL.
