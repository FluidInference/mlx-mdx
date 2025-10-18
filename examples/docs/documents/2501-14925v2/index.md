---
title: 2501.14925v2
source_url: file:///Users/brandonweng/code/mlx-mdx/examples/2501.14925v2.pdf
retrieved_at: 2025-10-18T05:27:23Z
---
# Profiling Apple Silicon Performance for ML Training

Dahua Feng\*
University of Virginia
wwh8us@virginia.edu

Rongxiang Wang
University of Virginia
waq9hw@virginia.edu

Zhiming Xu\*
zhiming.xu@gmail.com

Felix Xiaozhu Lin
University of Virginia
felixlin@virginia.edu

---

## ABSTRACT

Apple Silicon has attracted much attention for its performance and role in machine learning (ML) training. Unlike NVIDIA GPUs, which have traditionally dominated ML training, Apple Silicon has a significant difference in memory architecture. It uses Unified Memory, which integrates CPU and GPU memory instead of separate CPU memory and GPU VRAM. However, it is difficult to tell whether Unified Memory means more performance benefits.

This paper investigates the performance differences by training several large language model (LLM) workloads end-to-end under different memory scenarios. The results show a significant performance gap between Apple Silicon and NVIDIA GPUs. This paper attributes this gap to system-level factors such as page faults, power consumption, and kernel launch time. In addition, the performance difference of basic linear algebra subprograms (BLAS) on the NVIDIA GPUs and Apple Silicon chips is analyzed to further explain the observed gap.

GPU constraints, offering a portable, cost-effective alternative to traditional GPU workstations, and democratizing ML training and research. Besides, software environment support for Apple Silicon is also constantly improving. Pytorch has stable support for the MPS backend since version 1.12 [1], which provides great support for using Apple Silicon for machine learning tasks. Later, the introduction of the MLX framework [4] further optimized the performance of Apple Silicon in machine learning. However, despite these promising advantages, Apple Silicon devices still face scrutiny for computing power, especially when training LLMs, which require high throughput and large amounts of memory bandwidth. It must be admitted that the performance of Apple Silicon devices during training is poor compared to NVIDIA GPUs, but the reasons for the performance gap are not clear, which has brought resistance to the popularity of Apple Silicon in LLM training.

This paper focuses on the scenario under the single-chip LLM training and fine-tuning, explores the potential and limitations of Apple Silicon, and studies its architectural advantages, memory management capabilities, and performance trade-offs. Ultimately, our goal is to evaluate whether Apple devices like MacBooks can provide a viable and accessible solution for ML training, thereby bridging the gap between professional hardware and consumer-level options. We mainly answer the following questions in this paper: (1) What are the specific advantages and disadvantages of Apple Silicon and NVIDIA GPUs in LLM training? (2) What may be the reasons for these advantages and disadvantages of Apple Silicon? (3) How can Apple Silicon be used for LLM training to achieve more satisfactory performance?

## 1 INTRODUCTION

The rapid growth of large language models (LLMs) has expanded machine learning research while introducing significant computational and memory challenges. As LLM sizes increase, their memory requirements often surpass the VRAM limits of high-end GPUs, complicating large-scale training on standard GPU setups. This typically necessitates complex scheduling algorithms, reducing training efficiency. Consequently, access to sufficient training resources has become a major hurdle, particularly for independent researchers and small institutions.

Apple Silicon marks a shift with its M1 and M2 chips, unifying CPU, GPU, and Neural Engine in a unified memory pool of up to 128GB. This architecture addresses VRAM-limited

\* The first author and second author contribute equally to this work.

---

Conference’17, July 2017, Washington, DC, USA
2025. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM...$15.00
https://doi.org/10.1145/nnnnnnn.nnnnnnn


<!-- page 1 end -->


Apple’s Neural Engine, into a single unified memory architecture, which is a notable feature of Apple Silicon. This design not only simplifies data processing but also supports shared memory between processing units, significantly reducing latency and energy consumption compared to traditional multi-chip setups.

The unified memory architecture in Apple Silicon is particularly exciting under the context of LLM training. Unlike discrete GPUs that rely on limited capacity dedicated VRAM, Apple Silicon’s unified memory provides a more flexible and richer memory pool. Devices like MacBook Pro can support up to 128GB of unified memory, and Mac Studio can support up to 192GB of unified memory, which will be very beneficial for training tasks that require a lot of memory to handle large-scale LLMs. However, the efficiency and memory pooling advantages of the unified memory can be offset by lower raw compute throughput compared to dedicated GPU workstations, which limits Apple Silicon’s competitiveness in large-scale or production-level machine learning workflows. This trade-off sets the stage for a deeper exploration of Apple Silicon’s capabilities and limitations, evaluating its viability as a viable platform for accessible machine learning training, especially in the context of LLM and other resource-demanding models.

## 2.2 Software Support

**Pytorch.** With the release of Apple Silicon, Apple introduced the Metal Performance Shaders [9] framework, i.e., MPS, a high-performance, low-level API optimized for GPU-accelerated computing on macOS devices. The MPS backend enables PyTorch to take advantage of Apple’s GPU architecture for efficient memory handling and computational acceleration, making it possible to perform complex machine learning inference or training tasks on devices that were previously limited to CPU processing on macOS. Unlike traditional GPU architectures designed around independent VRAM, Apple Silicon’s unified memory architecture provides a shared memory pool accessible to CPU and GPU cores, which may have advantages when managing large LLM models that require a lot of memory capacity. The support for the MPS backend is still being improved. Currently, Pytorch still lacks support for some training-related optimizations, which are already very complete on the support for CUDA devices.

**MLX.** MLX is an array framework for efficient machine learning on Apple Silicon supporting Apple machine learning research [4]. Specifically, MLX can perform common machine learning operations such as matrix multiplication, convolution, and data transformations faster, which are critical for training and inference in deep learning models. In the software stack, MLX is located in the same position as Pytorch, which means MLX is potentially an alternative for Pytorch in machine learning on the Apple Silicon. The most significant feature of MLX is its unified memory model, which is also a feature that distinguishes it from the PyTorch with MPS above. Arrays in MLX exist in shared memory, allowing MLX arrays to be operated across supported device types without the need for data transfer. We take the MLX in the scope, with the hope that MLX can have a difference on machine learning using Apple Silicon and we can have a better understanding of the influence of software support on machine learning training performance.

## 3 METHOD

### 3.1 Choices of hardware

We use several servers and devices to conduct our experiments. We mainly focus on devices with comparable costs that can be afforded by regular research groups and institutions. For Nvidia devices, we conduct experiments on servers with four distinct models of middle- to high-end consumer and professional GPUs. For Apple Silicon devices, we conduct experiments on three M2 series models. Table 1 shows the size of VRAM/Unified Memory, the single-precision computing power, and the price of each device [3, 7, 8]. We do not include RTX A100 or similar products popular in data centers because of their limited availability, especially when ordered in smaller quantities, and prohibitively high prices.

---

**Table 1: Related parameters of each device.**

| Device | RAM(GB) | Computing power(TFLOPS) | Launch price(USD) |
|---|---|---|---|
| Quadro RTX 4000 | 8 | 7 | 899 |
| NVIDIA RTX 2080Ti | 11 | 13 | 999 |
| NVIDIA GeForce RTX 4090 | 24 | 83 | 1599 |
| NVIDIA RTX A6000 | 48 | 39 | 4650 |
| M2 Ultra | 192 | 27 | 6599 |
| M2 Max | 64 | 14 | 2599 |
| M2 Pro | 32 | 7 | 1999 |

---

**Figure 1: The software/hardware stack for ML training, showing for both Apple GPU and NVIDIA GPU.**
> Figure insight: Figure 1: The software/hardware stack for ML training, showing for both Apple GPU and NVIDIA GPU.

ML Workloads

Pytorch Backend MLX framework

CUDA MPS

NVIDIA GPU Apple Silicon


<!-- page 2 end -->


```mermaid
graph LR
    A[NVIDIA GPU VRAM] --> B[Scenario 1: Both CUDA and Mac devices have enough memory]
    A --> C[Scenario 2: CUDA device has insufficient memory, but Mac device has enough memory]
    A --> D[Scenario 3: Mac device has enough memory but quite near the capacity while CUDA device has insufficient memory]
    B --> E[Apple Silicon Unified Memory]
    C --> E
    D --> E
```

used in pretraining in addition to merely fine-tuning. We experiment on both pretraining and finetuning the large and XL variations of GPT-2.

We evaluated the model pre-training and fine-tuning under different settings. When all of the model and the training data are stored in VRAM or memory, the training process becomes memory-constrained. If the memory is not sufficient, ZeRO-Offload [17], which can offload part of the model to the CPU or DRAM, etc., needs to be applied to ensure the training can be processed smoothly. We list the total parameters and theoretical memory consumption of each model in Table 2. In practice, the runtime memory footprint is slightly larger due to the activations and so on. Depending on the total available memory on the device we evaluate, these workflows fall under different scenarios as introduced in Figure 2. We outline the corresponding memory scenario for each device and workload combination in Table 3. We adjust the sizes under each memory scenario on all the devices. They are as large as possible for the respective scenario.

We select the time required for a single pass on the same, fixed batch size during training and the memory usage during training as the metrics for our experiments. In particular, for Apple Silicon, we also observe the number of page faults and try to better understand how the unified memory impacts the performance. For energy efficiency, we measure the energy consumption of the Nvidia GPUs, and the GPUs of Apple silicon SoCs.

## 4 FINDINGS

### 4.1 End-to-end evaluations

**Precision** For both CUDA and Mac devices, we select FP32 as training precision. In addition, we try the mixed precision training [14], which is marked as AMP.FP16 and AMP.BF16, respectively, for CUDA only because at this time the torch AMP lacks support for Apple silicon SoCs.

**Scenario 1: Both CUDA and Apple silicon devices have enough memory** In this scenario, the model and training data can both fit into the Nvidia GPU VRAM and the unified memory of Apple silicon SoC.

In the first benchmark, we pre-train the medium checkpoint of the Whisper model on the Hugging Face platform [19], on CUDA and Mac devices, with the Common Voice dataset [2]. Figure 3(a)(1) shows the forward and backward time. In the second benchmark, we fine-tune the checkpoint of the GPT2-large model on the Hugging Face platform, on the IMDB dataset [12], with LoRA [5] and without LoRA respectively. Figure 3(a)(2) and (3) show the forward time and backward time without and with LoRA respectively.

We have to admit that all Apple silicon SoCs underperform Nvidia GPUs under this circumstance. The most significant difference between the two kinds of devices during training

---

**Table 2: The workloads we used in our experiments, with AdamW optimizer.**

| Workload | Parameters(M) | Theoretical Memory (GiB) |
|----------|---------------|--------------------------|
| Whisper-medium | 769 | 12.016 |
| Whisper-large | 1550 | 24.219 |
| GPT2-large | 774 | 12.094 |
| GPT2-XL | 1558 | 24.344 |

---

**Table 3: The configuration of our experiments for end-to-end training on the Hugging Face platform.**

| Scenario | Workload | Batch Size | Testbed |
|----------|----------|-----------|---------|
| 1        | Whisper-medium | 4 | NVIDIA GeForce RTX 4090 |
|          | Whisper-medium | 4 | NVIDIA RTX A6000 |
|          | Whisper-medium | 4 | M2 Ultra |
|          | Whisper-medium | 4 | M2 Max |
|          | GPT2-large | 16 | NVIDIA GeForce RTX 4090 |
|          | GPT2-large | 16 | NVIDIA RTX A6000 |
|          | GPT2-large | 16 | M2 Ultra |
|          | GPT2-large | 16 | M2 Max |
|          | GPT2-large | 16 | M2 Pro |
| 2        | Whisper-medium | 4 | Quadro RTX 4000 |
|          | Whisper-medium | 4 | NVIDIA RTX 2080Ti |
|          | GPT2-XL | 16 | NVIDIA GeForce RTX 4090 |
|          | Whisper-large | 32 | M2 Max |

---

### 3.2 Methodologies

In our experiment, we select a representative set of the most widely used workloads in the training and fine-tuning of large models. Specifically, we conduct experiments on two modalities of data, i.e., speech and text.

For speech, we choose Whisper, an automatic speech recognition model designed to transcribe spoken words into text with high accuracy for multiple languages and dialects [15]. Based on the Transformer architecture [18], Whisper is computationally intensive, requiring a lot of memory and processing resources to manage its large number of parameters and operate efficiently on complex audio data. Due to the model complexity, we pretrain the medium variation of Whisper.

For text, we both choose GPT-2 [16], a classic large language model that precedes Whisper but is built on the same Transformer architecture. The computational complexity and memory footprint of GPT-2 is slightly less than that of Whisper, making it a great example of examining whether less computationally capable Apple Silicon devices can be


<!-- page 3 end -->


is memory usage. On all CUDA devices, memory usage is relatively stable, which means that NVIDIA GPUs transfer almost all required data to VRAM at once during training; on Apple Silicon, memory usage(RSS) increases gradually, which means that Apple Silicon gradually transfers required data to Unified Memory during training.

In addition, we pre-train the GPT2-large with the MLX framework on M2 Ultra, M2 Max, and M2 Pro respectively, using the PTB corpus dataset [13]. Unlike the MPS backend, in this setting, the RSS remains stable. For a fair comparison, we furthermore pre-train the GPT2-large with Pytorch MPS or CUDA backend only (i.e. without Hugging Face platform) on M2 Ultra, M2 Max, M2 Pro, and RTX A6000 respectively on the PTB corpus dataset. The time required for each pass with two frameworks is shown in Table 4, which reveals that the MLX framework has a better performance than the MPS backend on Transformer-based neural network training. Also, in this case, the RTX A6000 slightly underperforms compared to the M2 Ultra(MLX).

**Scenario 2: CUDA device has insufficient memory but Mac device has enough memory** In this scenario, the model and training data require more memory than the capacity of Nvidia GPU VRAM. When the CUDA device has insufficient memory, in order to ensure that training can proceed normally, we must offload part of the data to the CPU or DRAM, which will bring additional data transmission overhead to the training process. We fine-tune the medium checkpoint of the Whisper model on both CUDA and Mac devices and use the Zero-Offload algorithm to complete the offload operation on CUDA devices to ensure the training process can be completed. The forward and backward time have been reported in Figure 3(b)(1). We also conduct the experiment on the workload of the XL checkpoint of the GPT2 model without LoRA, and the result is also in Figure 3(b)(2). Through experiments, we can see that the overhead of the data transmission required by ZeRO-Offload on CUDA devices is quite huge. For instance, even on NVIDIA GeForce RTX 4090, if ZeRO-Offload is applied, the training performance is still worse than M2 Ultra.

**Scenario 3: Mac device has enough memory but quite near the capacity** Moreover, we pre-train the large checkpoint of the Whisper model and adjust the batch size to 32 on M2 Max. In this scenario, the Mac device still has enough memory, but the total required memory is very approaching the device capacity. We observe the changes in Figure 4 during near capacity training and find that RSS is still increasing, but the growth rate is significantly slower than that when memory is super sufficient, and the RSS has maintained at a lower level than the model size. The time required for each pass in Figure 4 is also observed as a great performance impact and unsteadiness compared to the scenario when the memory is sufficient.

---

**Table 4: The time of each pass during GPT2-large pre-training without the Hugging Face platform.**

| Testbed | Workload | Batch Size | MLX/CUDA One Pass Time(s) | MPS One Pass Time(s) |
|---|---|---|---|---|
| M2 Ultra | | 8 | 2.710 | 3.239 |
| M2 Max | GPT-large | 4 | 2.924 | 3.645 |
| M2 Pro | | 2 | 8.692 | 10.704 |
| A6000 | | 8 | 2.771 | - |

---

**Figure 3: The forward time, backward time on different devices on the two workloads.**
> Figure insight: Figure 3: The forward time, backward time on different devices on the two workloads. Table 4: The time of each pass during GPT2-large pre-training without the Hugging Face platform. | Testbed | Workload | Batch Size | MLX/CUDA One Pass Time(s) | MPS One Pass Time(s) | |---|---|---|---|---| | M2 Ultra | | 8 | 2.710 | 3.239 | | M2 Max | GPT-large | 4 | 2.924 | 3.645 | | M2 Pro | | 2 | 8.692 | 10.704 | | A6000 | | 8 | 2.771 | - | is memory usage. On all CUDA devices, memory usage is relatively stable, which means that NVIDIA GPUs transfer almost all required data to VRAM at once during training; on Apple Silicon, memory usage(RSS) increases gradually, which means that Apple Silicon gradually transfers required data to Unified Memory during training. In addition, we pre-train the GPT2-large with the MLX framework on M2 Ultra, M

(a) All the devices have sufficient memory.

(1) Pretrain Whisper-medium with the batch size of 4.
(2) Fine-tune (without LoRA) GPT2-large with the batch size of 16.
(3) Fine-tune (with LoRA) GPT2-large with the batch size of 16.

(b) CUDA devices are applied with ZeRO-Offload.

---

**Figure 4: Memory consumption over time during near-capacity training of Whisper-large on M2 Max.**
> Figure insight: Figure 4: Memory consumption over time during near-capacity training of Whisper-large on M2 Max. Looking at the above scenarios together, we can draw the following conclusions: (1) Apple Silicons underperform the

---

Looking at the above scenarios together, we can draw the following conclusions: (1) Apple Silicons underperform the


<!-- page 4 end -->


other NVIDIA GPUs when the memory footprint required is lower than the capacity of NVIDIA GPUs; (2) Apple Silicons have a better performance when the VRAM of GPUs is not enough and ZeRO-Offload must be applied.

4.2 Energy efficiency
We also measure and record the GPU energy consumption per iteration of each device during the training the medium checkpoint of the Whisper model on the Common Voice dataset with the batch size of 4 and data type of FP32 (in Scenario 1).

In terms of the energy efficiency of the training, we observe that Apple Silicon demonstrates superior energy efficiency compared to other hardware platforms. As shown in the measurements shown in Table.5, the energy consumption per iteration of two Apple Silicon devices is similar to RTX 4090 and much better than A6000. Moreover, considering that the Nvidia GPU platforms need additional power supply for their CPU and memory, which will usually be about 70W [11], the gap would be further widened. The improvement in power efficiency not only reduces costs in training but also aligns with sustainable practices by reducing total energy consumption.

4.3 System issues
To further explain the performance gap in end-to-end training, we will move to the system level.

Memory usage over time As mentioned before, Apple Silicons seem to have a greatly different mechanism of memory management from NVIDIA GPUs. Regardless of the workload, Apple Silicon’s memory consumption during training shows a gradual increase, rather than maintaining relatively stable like NVIDIA GPUs, which may be a major reason for the underperformance of Apple Silicon in LLM training.

Page faults over time To get a better understanding of the overhead of the memory management of Apple Silicon during the training process, we measure the number of page faults of M2 Max when we pre-train the medium checkpoint of the Whisper model in Scenario 1. With the iteration of passes, the number of page faults is increasing gradually, which indicates that Mac triggers the page fault continuously in an attempt to transfer some data into memory. We also measure the number of page faults of M2 Max during the large checkpoint of the Whisper model pre-training with the batch size of 32 in Scenario 3. In this case, the actual memory required is near the SoC capacity, and more page faults are triggered with a stronger growth trend.

Overhead of the GPU runtime We conduct the experiment to measure the latency of the kernel launch on CUDA devices and Apple devices respectively, which is the layer below the BLAS and MLX. Table 6 shows the kernel launch time during the multiple iterations. The "cold launch" (i.e. the first time launching a kernel) is slower than the following launch and CUDA devices get less latency than Apple Silicon chips. Then when we repeatedly launch the kernel, the latency on CUDA devices is still much lower than that on Apple Silicon. Some common belief may be that Apple Silicon has a more optimized kernel launch time thanks to the unified memory architecture; however, our measurement shows the contract results. Besides, we do not observe any thermal-related issues on Apple silicon devices.

Table 5: The GPU energy consumption per iteration of each device during the training.

| Testbed | M2 Ultra | M2 MAX | A6000 | RTX 4090 |
|---|---|---|---|---|
| Energy per Iteration(J) | 114.86 | 129.30 | 212.39 | 108.58 |

Table 6: Kernel launch time on different devices.

| Testbed | Cold Launch Time(ms) | Follow-up Launch Time(ms) |
|---|---|---|
| M2 Ultra | 0.7859 | 0.1480 |
| M2 Max | 0.9080 | 0.1273 |
| M2 Pro | 0.5850 | 0.2637 |
| A6000 | 0.0225 | 0.0064 |
| RTX 4090 | 0.0225 | 0.0036 |
| 2080Ti | 0.0845 | 0.0053 |
| RTX 4000 | 0.1750 | 0.0046 |

Figure 5: Measurements on Matrix-Matrix product.
> Figure insight: The figure shows the throughput (ops/sec) for Matrix-Matrix product on different testbeds. The throughput is higher for NVIDIA GPUs (FP32) than for Apple Silicon GPUs (FP16). The throughput for NVIDIA GPUs decreases with the batch size, while for Apple Silicon GPUs, the throughput remains relatively stable.

4.4 BLAS kernel analysis
To explain the performance gap, we dive into basic linear algebra subprograms (BLAS) performance differences between NVIDIA GPUs and Apple Silicon GPUs. Since the majority of operations of neural networks are on matrices, BLAS performance largely determines the aforementioned performance of end-to-end training [10].

We conduct the experiments on Matrix-Matrix Product, Matrix-Vector Product, and Vector-Jacobian Product respectively across a variety of shapes. Additionally, we benchmark both the FP16 and FP32 data types. Based on computing power, we compare the computing throughput of A6000 and M2 Ultra, 2080Ti and M2 Max, and RTX4000 and M2 Pro. We use both MPS in Pytorch and MLX framework for Apple Silicon and CUDA in Pytorch for NVIDIA GPUs as the


<!-- page 5 end -->


software support. As we mentioned in 4.3, the cold launch will have an apparently longer latency than the follow-up launch. Thus, in order to eliminate the impact that the cold launch may cause, we first warm up for 10 iterations and then launch the kernel 100 times continuously and repeatedly and report the average of all these results to reduce the effect of unsteadiness.

**Matrix-Matrix Product** We benchmark Batched Matrix-Matrix Product for the batch size of 8 and matrix shapes of (2000\*2000), which are common in latest transformer-based LLMs. The benchmark results are shown in Figure 5.

NVIDIA GPUs, which have support for Tensor Cores, can get more acceleration when using FP16 instead of FP32, while MPS can hardly get any benefits and MLX can get only about 20%-30% benefit. With FP16, MLX underperforms under all the conditions compared to CUDA but has a better performance than MPS in most cases. This may be because the Tensor Cores of NVIDIA GPUs have very good optimized support for FP16-type operations, and this support is not yet perfect in the Apple Silicon ecosystem. With FP32, the performance gap between MLX or MPS and CUDA is not as large as that with FP16. MLX and MPS shows slightly weaker but largely comparable performances as CUDA.

**Matrix-Vector Product** We also benchmark the Matrix-Vector Product operations with various shapes of matrices and vectors. These operations are also common in transformer-based LLMs, representing some of the small operations when memory bandwidth may be a bottleneck. The benchmark results are shown in Figure 6.

On both CUDA and Apple Silicon devices, using FP16 has a similar ratio of acceleration compared to using FP32 when the shape is relatively large, and using FP16 has similar performance to using FP32 for some small matrices and vectors. For Matrix-Vector Product, Tensor Cores cannot optimize the computation as well as for Matrix-Matrix Product. Additionally, Matrix-Vector Product usually involves smaller data sizes, often hitting memory bandwidth limits rather than computational limits. Compared with FP16, the performance gap between MLX and CUDA has narrowed, and MLX even outperforms CUDA in some cases. MPS shows a similar performance in the measurement compared to MLX.

**Vector-Jacobian Product** We conduct benchmarking of Vector-Jacobian Product operations, which are the main operations during the backpropagating process, across various shapes of matrices and vectors. The benchmark results are shown in Figure 7.

In CUDA cases, using FP16 can obtain an obvious (about 5x-6x) acceleration compared to using FP32; but in MLX or MPS cases, using FP16 can reach about 2x acceleration. With FP32, MLX performs better than CUDA if the shapes are relatively large, but they have similar performance if tensors are in small shapes; with FP16, CUDA performs better than MLX if the shapes are relatively large, but still have similar performance if tensors are in small shapes. However, for the Vector-Jacobian Product kernel, MPS shows an obvious underperformance compared to MLX.

**Conclusion:** The performance gap observed in BLAS kernels (primary factors), combined with the system issues explored earlier (secondary factors), can explain the performance gap we observed for the end-to-end training. Therefore, we believe we have identified the root causes.

---

**Figure 6: Measurements on Matrix-Vector product.**
> Figure insight: The figure shows the performance of different GPU architectures (MLX, CUDA, MPS) for Matrix-Vector Product operations with various matrix shapes and data types. Notable trends include the performance gap between MLX and CUDA narrowing with FP16, and MPS showing similar performance to MLX.

(a) Comparison between A6000 and M2 Ultra.
(b) Comparison between RTX 2080Ti and M2 Max.
(c) Comparison between RTX 4000 and M2 Pro.

---

**Figure 7: Measurements on Vector-Jacobian Product.**
> Figure insight: Figure 7: Measurements on Vector-Jacobian Product. Software support. As we mentioned in 4.3, the cold launch will have an apparently longer latency than the follow-up launch. Thus, in order to eliminate the impact that the cold launch may cause, we first warm up for 10 iterations and then launch the kernel 100 times continuously and repeatedly and report the average of all these results to reduce the effect of unsteadiness. Matrix-Matrix Product We benchmark Batched Matrix-Matrix Product for the batch size of 8 and matrix shapes of (2000*2000), which are common in latest transformer-based LLMs. The benchmark results are shown in Figure 5. NVIDIA GPUs, which have support for Tensor Cores, can get more acceleration when using FP16 instead of FP32, while MPS can hardly get any benefits and MLX can get only about 20%-30% benefit. With FP16, MLX underperforms under all the conditions compared to CUDA but has a better performance than MPS in most cases. This may be because the Tensor Cores of NVIDIA GPUs have very good optimized support for FP16-type operations, and this support is not yet perfect in

(a) Comparison between A6000 and M2 Ultra.
(b) Comparison between RTX 2080Ti and M2 Max.
(c) Comparison between RTX 4000 and M2 Pro.


<!-- page 6 end -->


larger than the memory footprint of training the model, going ahead and using it will not be too terrible considering that it may cost much more if you buy the NVIDIA GPU cards that can fit the model size, but with an expectation that the speed is about 3x-4x lower than an NVIDIA GPU of similar price. If you want to purchase a new machine dedicated to machine learning training, then Mac devices will be a better choice only if you are about to dive into larger models, but for most general models, NVIDIA GPU will always be a smart choice. From our observation, thermal is not a big concern. The main concern should be the slow speed when Mac devices compute with FP16 data type.

### 5.2 For hardware/software vendors

Optimized kernels for Apple Silicon achieve only a 30%-40% speed advantage over PyTorch, and even after fully porting MLX to PyTorch, a significant performance gap still exists. To make Apple Silicon a more viable and attractive option for training, a road map must address key areas. First, introducing dedicated hardware for FP16 operations, such as Tensor Cores, could significantly enhance performance, though a strong justification for the necessary hardware investments would be required. Next, expanding FP16 support within Automatic Mixed Precision(AMP) would improve efficiency and usability for mixed precision training. Finally, ensuring seamless integration of these advancements into the Apple Silicon training ecosystem would be essential to realize their full potential.

### 5.3 For ML benchmark development

For future tests on Apple Silicon, the training performance with FP16 should be checked; if the software support for FP16 remains lacking, the performance gap will likely persist.

### 6 CONCLUDING REMARKS

This paper has measured the performance of LLM training on the Apple Silicon and made comparisons side-by-side with some NVIDIA GPUs with similar prices. We also try to explain the performance gap from the system level and dive into BLAS kernels to figure out where the gap is from. As a result, based on our observation, we have some recommendations from various perspectives.

### REFERENCES

[1] Previous PyTorch Versions – pytorch.org. https://pytorch.org/get-started/previous-versions/, 2024. [Accessed 19-12-2024].

[2] Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M. Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus, 2020.

[3] Nvidia Corporation. Nvidia Marketplace. https://marketplace.nvidia.com/en-us/enterprise/laptops-workstations/, 2024. [Accessed 14-12-2024].

[4] Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. MLX: Efficient and flexible machine learning on apple silicon, 2023.

[5] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

[6] Apple Inc. Apple unleashes m1. https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/, 2020. [Accessed 14-12-2024].

[7] Apple Inc. Apple introduces m2 ultra. https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/, 2023. [Accessed 14-12-2024].

[8] Apple Inc. Apple unveils m2 pro and m2 max: next-generation chips for next-level workflows. https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/, 2023. [Accessed 14-12-2024].

[9] Apple Inc. Accelerated pytorch training on mac. https://developer.apple.com/metal/pytorch/, 2024. [Accessed 14-12-2024].

[10] Jeremy Kepner, Manoj Kumar, José Moreira, Pratap Pattnaik, Mauricio Serrano, and Henry Tufo. Enabling massive deep neural networks with the graphblas. In 2017 IEEE High Performance Extreme Computing Conference (HPEC), pages 1–10. IEEE, 2017.

[11] Adam Lewis, Soumik Ghosh, and N.-F. Tzeng. Run-time energy consumption estimation based on workload in server systems. In Proceedings of the 2008 Conference on Power Aware Computing and Systems, HotPower’08, page 4, USA, 2008. USENIX Association.

[12] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Dekang Lin, Yuji Matsumoto, and Rada Mihalcea, editors, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics.

[13] Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313–330, 1993.

[14] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training, 2018.

[15] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. arxiv 2022. arXiv preprint arXiv:2212.04356, 10, 2022.

[16] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[17] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. {Zero-offload}: Democratizing {billion-scale} model training. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 551–564, 2021.


<!-- page 7 end -->


[18] A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017.
[19] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface’s transformers: State-of-the-art natural language processing, 2020.
