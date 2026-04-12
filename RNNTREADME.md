# Whisper ASR (Hulk Framework) RNNT Project Technical Guide

This document provides a comprehensive technical overview of the Whisper ASR project based on the **Hulk** framework, focusing on the **RNNT (Transducer)** architecture. It serves as a guide for implementing new languages (e.g., Hungarian) and understanding the existing French pipeline.

---

## 1. System Architecture

The project utilizes a Multi-Task Learning (MTL) strategy combining RNNT with CTC and CE branches for optimal performance and stability.

### 1.1 Core Components
- **Encoder (Acoustic Encoder)**: Extracts high-level acoustic features from 40-dim Fbank input.
- **Predictor (Language Model)**: Predicts the next subword (BPE) based on previous history.
- **Joiner (Joint Network)**: Fuses Encoder and Predictor outputs to predict final vocabulary probabilities.

### 1.2 Multi-Task Learning (MTL) Branches
- **Phone-CE Branch**: Integrated into intermediate Encoder layers, supervised by Phone-level labels via Force Alignment (FA).
- **BPE-CTC Branch**: Attached to the top of the Encoder, supervised by Subword (BPE) labels to provide monotonic alignment constraints.

---

## 2. Data Preparation Pipeline

### 2.1 Pfile Data Format
Pfile is the binary container format optimized for large-scale ASR training.
- **Header**: Stores metadata (dimensions, frames, sentence count).
- **Data**: Interleaved features and labels (Big-Endian storage).
- **Tail Index**: Byte offsets for each sentence, enabling fast random access.

### 2.2 Labeling Levels
- **CE Labels (Phone-level)**: Generated via **Force Alignment (FA)**. Essential for pre-training the acoustic encoder.
- **ED/CTC Labels (Subword-level)**: 
  - Generated using **BPE (Byte Pair Encoding)** via SentencePiece.
  - **Constraint**: `<blank>` must be fixed at **ID 0** to satisfy RNNTLoss and CTC logic.
  - **Hungarian Support**: Must include accented characters: `á, é, í, ó, ö, ő, ú, ü, ű`.

### 2.3 Synchronization & Filtering
- **Tooling**: `11.get_same_labpfile.pl` & `12.run_pfile_rand_by_index.sh`.
- **Logic**: Filters out sentences where feature frames do not match label frames (critical for CE training).

---

## 3. Training Evolution Strategy

The training follows a "Pre-train -> Fine-tune -> Clamp -> Quantize" lifecycle:

### Step 10: Cross-lingual Transfer (Russian/French Init)
- **Goal**: Initialize weights from a mature model (e.g., Russian) to accelerate convergence on a new language.
- **Focus**: Intermediate CTC/CE training to establish basic acoustic understanding.

### Step 11: Acoustic Foundation (CE Training)
- **Goal**: Train the Encoder to recognize phonemes (Phone Set) accurately.
- **Focus**: High-precision frame-level supervision.

### Step 11_cectc: Joint RNNT Training
- **Goal**: Train the end-to-end RNNT model while keeping CTC/CE as auxiliary losses.
- **Logic**: `loss_binary` trains the model's confidence in predicting `blank`, preparing for "Frame Skipping" during inference.

### Step 11_clamp & clamp_2: Numerical Constraint (QAT)
- **Goal**: Prepare for hardware deployment (NPU/DSP) by limiting the dynamic range of weights and activations.
- **Mechanism**: `torchintx.clamp_module` truncates values (e.g., Bias to 1.5, Outputs to 4.0).

### Step 12: Quantization Analysis (PTQ)
- **Goal**: Search for optimal quantization scale factors.
- **Output**: An 8-bit quantized model ready for on-device deployment.

---

## 4. Advanced Optimizations

### 4.1 Frame Skipping Mechanism
- **Logic**: If the Skip Predictor (`logits_enc_binary`) predicts a `blank` probability > 0.9, the Decoder and Joiner computations are bypassed.
- **Benefit**: 2-3x speedup in inference latency on edge devices.

### 4.2 BPE-CTC Integration
- **Role**: CTC provides a strong monotonic alignment constraint to the "greedy" RNNT, preventing alignment drift and improving stability on long utterances.

---

## 5. Implementation Checklist for New Languages (e.g., Hungarian)

1.  **Text Cleaning**: Use `get_sent_mlf.py` with the HU alphabet; handle `õ` -> `ő` encoding issues.
2.  **SPM Training**: Run `train_hu_spm.py` (BPE 5k, coverage 0.9995, ID 0 = `<blank>`).
3.  **Pfile Packaging**: Prepare both Phone-level (FA) and BPE-level labels.
4.  **MTL Training**: Start from a pre-trained encoder (e.g., French Step 11) and fine-tune with RNNT + CTC losses.
