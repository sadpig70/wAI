# EEG-wAI: Brain Wave-based AI System
## Real-time Neural Signal Decoding and Neurofeedback Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenBCI Compatible](https://img.shields.io/badge/OpenBCI-Compatible-green)](https://openbci.com/)

---

## 🧠 Executive Summary

**EEG-wAI** is a specialized implementation of the wAI (Wave-based AI) framework focused on brain wave analysis. This proof-of-concept system transforms EEG signals into a "neural language" through advanced tokenization and grammar extraction, enabling real-time decoding of mental states, intentions, and emotions with closed-loop neurofeedback capabilities.

### Core Innovation
- **Neural Tokenization**: Transform continuous EEG into discrete semantic tokens
- **Real-time Processing**: Sub-200ms end-to-end latency for BCI applications
- **State Decoding**: Attention, relaxation, cognitive load, and emotional states
- **Closed-loop Feedback**: Adaptive neurofeedback for brain state optimization

---

## 🎯 Project Overview

### Objectives
1. Tokenize and grammaticalize brain waves as a structured language
2. Decode mental states (attention/relaxation) with >85% accuracy
3. Implement real-time feedback loops for brain state optimization
4. Establish foundation for advanced BCI applications

### Core Hypothesis
> Brain waves contain linguistic patterns that can be tokenized using state-space models (Mamba/SSM) and Chronos-style tokenization, enabling real-time BCI and closed-loop neurofeedback systems.

---

## 🔧 Technical Specifications

### Phase 1: Hardware & Sampling Configuration

#### Fixed Specifications (PoC)
| Component | Specification | Justification |
|-----------|--------------|---------------|
| **EEG Device** | OpenBCI Mark IV (Cyton + Daisy) | Cost-effective, research-grade, open-source |
| **Channels** | 16 channels | 10-20 system: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, Pz, Cz |
| **Sampling Rate** | 250 Hz | OpenBCI standard, sufficient for 0.5-45 Hz analysis |
| **Reference** | Common Mode Sense (CMS/DRL) | Impedance <10kΩ (target <5kΩ) |
| **Frequency Band** | 0.5-45 Hz | Covers δ, θ, α, β bands; Notch: 50/60 Hz |
| **Window/Hop** | 2.0s window (500 samples), 0.25s hop | Balance between temporal resolution and stability |
| **Latency Target** | <200ms (end-to-end, 95th percentile) | Real-time BCI requirement |
| **Feedback** | Audiovisual only (Phase 1) | Electrical stimulation deferred to Phase 2 (IRB required) |

#### Latency Budget Breakdown
```
Total: <200ms (95th percentile)
├── Acquisition Buffer: ≤50ms
├── Preprocessing: ≤25ms
├── Tokenization: ≤40ms
├── Inference: ≤50ms
├── Feedback Delivery: ≤25ms
└── Overhead: ≤10ms
```

### Phase 2: Upgrade Path
- **Enhanced Sampling**: 32 channels @ 1kHz (g.tec/BrainProducts)
- **Multimodal Integration**: PPG, EDA, respiration, EOG
- **Closed-loop Stimulation**: tACS/tDCS with safety protocols
- **Advanced Applications**: Motor imagery, language decoding

---

## 📡 Signal Processing Pipeline

### 1. Data Acquisition & Streaming

#### LSL Stream Configuration
```python
# EEG Stream Definition
eeg_stream = {
    "name": "EEG_PoC",
    "type": "EEG",
    "channel_count": 16,
    "nominal_srate": 250,
    "channel_format": "float32",
    "source_id": "OpenBCI_001"
}

# Marker Stream Definition
marker_stream = {
    "name": "EEG_Markers",
    "type": "Markers",
    "channel_count": 1,
    "channel_format": "string",
    "source_id": "TaskController_001"
}
```

### 2. Label Synchronization Protocol

#### Event Schema (Focus/Relaxation Task)
```python
EVENT_SCHEMA = {
    "READY": {"duration": 3.0, "description": "Preparation phase"},
    "FOCUS_ON": {"duration": 15.0, "task": "Stroop"},
    "FOCUS_OFF": {"duration": 0.0, "transition": True},
    "REST": {"duration": 10.0, "description": "Relaxation phase"},
    "REST_OFF": {"duration": 0.0, "transition": True}
}

# Marker Payload Format (JSON)
marker_example = {
    "event": "FOCUS_ON",
    "trial": 1,
    "task": "Stroop",
    "timestamp": 1234567890.123
}
```

#### Synchronization Requirements
- **Marker Latency**: ≤20ms (95th percentile)
- **Timing Jitter**: ≤±10ms (95th percentile)
- **Clock Drift**: <2ms per 30-minute session
- **Integrity Rules**: All trials must contain complete marker sequences

### 3. Real-time Processing Implementation

```python
import numpy as np
from scipy.signal import welch, stft
from collections import deque
from pylsl import StreamInlet, resolve_stream

class EEGProcessor:
    def __init__(self, fs=250, window_size=2.0, hop_size=0.25):
        self.fs = fs
        self.window_samples = int(window_size * fs)
        self.hop_samples = int(hop_size * fs)
        self.buffer = deque(maxlen=self.window_samples)
        self.bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30)
        }
        
    def process_window(self, data):
        """Process EEG window and extract features"""
        # data shape: (channels, samples)
        features = {
            'bandpower': self.compute_bandpower(data),
            'tfr_tokens': self.quantize_tfr(data),
            'emotion': self.estimate_emotion(data)
        }
        return features
    
    def compute_bandpower(self, data):
        """Extract band power features"""
        n_channels, n_samples = data.shape
        bandpowers = np.zeros((n_channels, len(self.bands)))
        
        for ch in range(n_channels):
            freqs, psd = welch(data[ch, :], fs=self.fs, nperseg=min(256, n_samples))
            
            for idx, (name, (low, high)) in enumerate(self.bands.items()):
                freq_idx = np.logical_and(freqs >= low, freqs < high)
                bandpowers[ch, idx] = np.trapz(psd[freq_idx], freqs[freq_idx])
                
        return 10 * np.log10(bandpowers + 1e-12)  # Convert to dB
    
    def quantize_tfr(self, data, n_bins=8):
        """Quantize time-frequency representation"""
        n_channels, n_samples = data.shape
        quantized = []
        
        for ch in range(n_channels):
            f, t, Zxx = stft(data[ch, :], fs=self.fs, nperseg=256, noverlap=192)
            magnitude = np.abs(Zxx)
            log_mag = np.log(magnitude + 1e-9)
            
            # Percentile-based quantization
            p_low, p_high = np.percentile(log_mag, [5, 95])
            normalized = np.clip((log_mag - p_low) / (p_high - p_low + 1e-6), 0, 1)
            quantized_ch = np.floor(normalized * (n_bins - 1)).astype(np.int8)
            quantized.append(quantized_ch)
            
        return np.stack(quantized, axis=0)
    
    def estimate_emotion(self, data):
        """Simple emotion estimation based on alpha/beta ratio"""
        bandpowers = self.compute_bandpower(data)
        alpha_power = np.mean(bandpowers[:, 2])  # Alpha band
        beta_power = np.mean(bandpowers[:, 3])   # Beta band
        
        # Higher alpha, lower beta indicates relaxation
        relaxation_score = np.tanh((alpha_power - beta_power) / 5.0)
        
        return {
            'relaxation': float(relaxation_score),
            'arousal': float(-relaxation_score)
        }
```

---

## 🤖 Machine Learning Architecture

### 1. Tokenization Framework

```python
class EEGTokenizer:
    """
    Transform continuous EEG into discrete tokens for sequence modeling
    """
    def __init__(self, vocab_size=1024, embedding_dim=256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.codebook = self._initialize_codebook()
        
    def _initialize_codebook(self):
        """Initialize VQ-VAE style codebook"""
        return np.random.randn(self.vocab_size, self.embedding_dim)
    
    def encode(self, features):
        """
        Convert features to token indices
        Input: Dictionary of features from EEGProcessor
        Output: Token sequence
        """
        # Flatten and concatenate features
        feature_vector = self._flatten_features(features)
        
        # Vector quantization
        distances = np.sum((feature_vector - self.codebook)**2, axis=1)
        token_idx = np.argmin(distances)
        
        return token_idx
    
    def decode(self, tokens):
        """Reconstruct features from tokens"""
        return self.codebook[tokens]
    
    def _flatten_features(self, features):
        """Flatten feature dictionary to vector"""
        flat = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                flat.append(value.flatten())
            elif isinstance(value, dict):
                flat.extend(value.values())
        return np.concatenate(flat)
```

### 2. State-Space Model Integration

```python
import torch
import torch.nn as nn

class EEGMamba(nn.Module):
    """
    Mamba-based state-space model for EEG sequence modeling
    """
    def __init__(self, d_model=256, d_state=64, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Token embedding
        self.embedding = nn.Embedding(1024, d_model)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state) 
            for _ in range(n_layers)
        ])
        
        # Classification heads
        self.attention_head = nn.Linear(d_model, 1)
        self.emotion_head = nn.Linear(d_model, 2)
        
    def forward(self, tokens):
        # Embed tokens
        x = self.embedding(tokens)
        
        # Process through Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Pool sequence
        x_pooled = x.mean(dim=1)
        
        # Predictions
        attention_score = torch.sigmoid(self.attention_head(x_pooled))
        emotion_scores = torch.softmax(self.emotion_head(x_pooled), dim=-1)
        
        return {
            'attention': attention_score,
            'relaxation': emotion_scores[:, 0],
            'arousal': emotion_scores[:, 1]
        }

class MambaBlock(nn.Module):
    """Simplified Mamba block for demonstration"""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.ssm = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        return self.norm(x + self.ssm(x))
```

---

## 📊 Performance Metrics & Validation

### Key Performance Indicators (KPIs)

| Metric | Target | Current | Method |
|--------|--------|---------|--------|
| **Attention Detection** | AUC ≥0.85 | Pending | Cross-validation on Stroop task |
| **Relaxation Classification** | F1 ≥0.80 | Pending | Alpha/Beta ratio validation |
| **End-to-end Latency** | <200ms (95p) | Pending | Automated timing logs |
| **Signal Quality** | SNR >10dB | Monitoring | Real-time impedance check |
| **User Calibration Time** | <5 minutes | Pending | Time to stable baseline |

### Validation Protocol

```python
class ValidationFramework:
    def __init__(self):
        self.metrics = {
            'auc': [],
            'f1': [],
            'latency': [],
            'snr': []
        }
        
    def validate_attention_detection(self, predictions, ground_truth):
        """Validate attention state classification"""
        from sklearn.metrics import roc_auc_score, f1_score
        
        auc = roc_auc_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions > 0.5)
        
        self.metrics['auc'].append(auc)
        self.metrics['f1'].append(f1)
        
        return {'auc': auc, 'f1': f1}
    
    def measure_latency(self, timestamps):
        """Calculate end-to-end latency"""
        latencies = []
        for ts in timestamps:
            latency = (ts['feedback'] - ts['acquisition']) * 1000  # ms
            latencies.append(latency)
            
        p95 = np.percentile(latencies, 95)
        self.metrics['latency'].append(p95)
        
        return {'p95_latency_ms': p95}
    
    def assess_signal_quality(self, eeg_data, noise_floor=1e-6):
        """Calculate signal-to-noise ratio"""
        signal_power = np.var(eeg_data)
        snr_db = 10 * np.log10(signal_power / noise_floor)
        
        self.metrics['snr'].append(snr_db)
        return {'snr_db': snr_db}
```

---

## 🧪 Experimental Protocols

### Minimum Viable Experiment 1: Focus vs. Rest Classification

```yaml
Objective: Binary classification of attention states
Participants: n=30 (pilot study)
Duration: 30 minutes per session

Protocol:
  - Baseline: 2 minutes eyes-closed rest
  - Trials: 20 repetitions of:
    - Ready: 3 seconds
    - Focus (Stroop task): 15 seconds
    - Rest: 10 seconds
  - Markers: LSL-synchronized events

Success Criteria:
  - AUC ≥0.85 for focus detection
  - Individual calibration <10 minutes
  - False positive rate <15%

Data Collection:
  - Raw EEG: 16 channels @ 250 Hz
  - Behavioral: Response time, accuracy
  - Subjective: NASA-TLX workload assessment
```

### Minimum Viable Experiment 2: Real-time Neurofeedback

```yaml
Objective: Closed-loop relaxation training
Setup: Visual/auditory feedback based on alpha power

Protocol:
  - Calibration: 5 minutes baseline
  - Training: 4 blocks × 5 minutes
    - Real-time alpha power feedback
    - Target: Increase alpha by 20%
  - Control: Sham feedback condition

Metrics:
  - Alpha power change pre/post
  - Learning curve across blocks
  - Subjective relaxation scores
  - Heart rate variability correlation
```

---

## 🛠️ Development Environment

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/EEG-wAI.git
cd EEG-wAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```python
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
mne>=0.24.0
pylsl>=1.14.0
torch>=1.10.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
psychopy>=2022.1.0  # For stimulus presentation
brainflow>=4.8.0    # Alternative to OpenBCI GUI
```

### Quick Start

```python
# run_experiment.py
from eeg_wai import EEGProcessor, EEGTokenizer, ValidationFramework
from pylsl import StreamInlet, resolve_stream
import time

def main():
    # Initialize components
    processor = EEGProcessor()
    tokenizer = EEGTokenizer()
    validator = ValidationFramework()
    
    # Connect to EEG stream
    print("Looking for EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    
    # Main loop
    print("Starting real-time processing...")
    while True:
        # Get data
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=64)
        
        if chunk:
            # Process
            features = processor.process_window(chunk)
            tokens = tokenizer.encode(features)
            
            # Feedback
            emotion = features['emotion']
            print(f"Relaxation: {emotion['relaxation']:.2f}")
            
        time.sleep(0.01)  # 10ms loop

if __name__ == "__main__":
    main()
```

---

## 🔒 Safety & Ethics

### Phase 1 Safety Protocols
- **No electrical stimulation**: Audiovisual feedback only
- **Data privacy**: No raw EEG storage, features only
- **Opt-in consent**: Clear data usage policies
- **Session limits**: Maximum 1 hour with breaks
- **Emergency stop**: Immediate termination capability

### Ethical Considerations
```python
class SafetyMonitor:
    """Monitor for adverse events and safety violations"""
    
    def __init__(self):
        self.thresholds = {
            'max_session_minutes': 60,
            'min_break_minutes': 10,
            'max_consecutive_trials': 50,
            'artifact_threshold': 0.3  # 30% bad channels
        }
        
    def check_session_duration(self, start_time):
        """Enforce maximum session duration"""
        duration = (time.time() - start_time) / 60
        if duration > self.thresholds['max_session_minutes']:
            return False, "Session time limit reached"
        return True, "OK"
    
    def check_signal_quality(self, impedances):
        """Monitor electrode impedances"""
        bad_channels = sum(z > 20 for z in impedances)  # 20kΩ threshold
        if bad_channels / len(impedances) > self.thresholds['artifact_threshold']:
            return False, "Poor signal quality"
        return True, "OK"
```

---

## 📈 Project Timeline

### Gantt Chart View
```
EEG-wAI_PoC                    [===============================] 100%
├── SpecFix (Hardware)         [==============] Complete
│   ├── Device Selection       [====] Complete
│   ├── Channel Config         [====] Complete
│   └── Latency Budget         [====] Complete
├── LabelSync (Protocol)       [==============] Complete
│   ├── LSL Setup             [====] Complete
│   ├── Event Schema          [====] Complete
│   └── Sync Validation       [====] Complete
├── TokenRef (Implementation)  [==============] Complete
│   ├── Feature Extraction    [====] Complete
│   ├── Tokenization          [====] Complete
│   └── Unit Tests            [====] Complete
├── MVP-1 (Focus/Rest)        [========------] In Progress
│   ├── Data Collection       [====] In Progress
│   ├── Model Training        [----] Pending
│   └── Validation            [----] Pending
└── Safety/Ethics             [==============] Ongoing
```

---

## 🤝 Contributing

We welcome contributions from neuroscientists, ML engineers, and BCI enthusiasts!

### Contribution Guidelines
1. **Code Style**: Follow PEP 8 for Python code
2. **Testing**: Include unit tests for new features
3. **Documentation**: Update relevant sections
4. **Safety First**: Any neural stimulation requires IRB approval

### Priority Development Areas
- [ ] Real-time artifact rejection algorithms
- [ ] Multi-subject transfer learning
- [ ] Mobile EEG device support
- [ ] WebAssembly deployment for browser-based BCI
- [ ] Integration with meditation/mindfulness apps

---

## 📚 References

### Key Papers
1. **EEG Decoding**: "Deep learning-based electroencephalography analysis: a systematic review" (2021)
2. **State-Space Models**: "Mamba: Linear-Time Sequence Modeling" (2023)
3. **BCI Systems**: "Brain-computer interfaces: current trends and applications" (2023)
4. **Neurofeedback**: "Real-time functional magnetic resonance imaging neurofeedback" (2022)

### Datasets
- **EEGBCI**: Motor Imagery dataset (PhysioNet)
- **DEAP**: Dataset for Emotion Analysis using EEG
- **Temple University EEG**: Large-scale clinical EEG

### Tools & Resources
- [OpenBCI Documentation](https://docs.openbci.com/)
- [MNE-Python](https://mne.tools/)
- [Lab Streaming Layer](https://labstreaminglayer.org/)
- [PsychoPy](https://www.psychopy.org/)

---

## 📧 Contact

- **GitHub Issues**: Bug reports and feature requests
- **Email**: eeg-wai@example.com
- **Discord**: [Join our BCI community](https://discord.gg/eeg-wai)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🎯 Vision Statement

> "EEG-wAI transforms brain waves from mysterious signals into a comprehensible language. By bridging neuroscience and artificial intelligence, we're not just reading minds – we're opening a bidirectional communication channel between human consciousness and digital intelligence. This is the foundation for a future where thought becomes action, intention becomes reality, and the boundaries between mind and machine gracefully dissolve."

**— The EEG-wAI Team**

---

*Version: 1.0.0 | Last Updated: January 2025*