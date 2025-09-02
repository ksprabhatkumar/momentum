# Momentum: Federated Learning for Human Activity Recognition

**Momentum** is a comprehensive, research-oriented framework for Human Activity Recognition (HAR) using Federated Learning. The project combines offline model training, distributed federated learning, real-time prediction streaming, and mobile data collection in a modular architecture designed for scalability and research flexibility.

## 🚀 Features

- **🤖 Advanced ML Models**: Temporal Convolutional Networks (TCN) optimized for HAR
- **🔗 Federated Learning**: Privacy-preserving distributed training across multiple clients
- **📱 Mobile Integration**: Native Android app for data collection and on-device inference
- **📊 Real-time Monitoring**: Live prediction streaming and comprehensive dashboards
- **⚡ Edge Deployment**: Quantized TFLite models for efficient mobile inference
- **📈 Research Tools**: Evaluation scripts, simulation environments, and visualization tools


## 🛠️ Core Components

### 1. 🎯 Model Training Pipeline
**Location:** `model_training/`

- **Temporal Convolutional Networks (TCN)** for sequential HAR data
- Automated data preprocessing and windowing
- Model export to Keras and quantized TFLite formats
- Comprehensive evaluation metrics and visualization
- Research-ready Jupyter notebooks for experimentation

**Key Features:**
- Multi-format model export (Keras, TFLite)
- Automated hyperparameter optimization
- Cross-validation and performance metrics
- Data augmentation and preprocessing pipelines

### 2. 🔗 Federated Learning Server
**Location:** `federated_server/src/server_deployment/`

- **Flask-based orchestration** for federated learning rounds
- Secure client aggregation using FedAvg algorithm
- Real-time metrics tracking and visualization
- RESTful API for model distribution and updates
- Interactive web dashboard for monitoring

**Key Features:**
- Multi-client coordination
- Secure aggregation protocols
- Live performance monitoring
- Model versioning and rollback
- Configurable aggregation strategies

### 3. 📊 Private Node (Real-time Streaming)
**Location:** `federated_server/src/private_node.py`

- **WebSocket-based streaming** for live predictions
- Real-time activity classification dashboard
- Low-latency prediction pipeline
- Multi-device connection support

**Key Features:**
- Real-time prediction visualization
- Device management and monitoring
- Prediction history and analytics
- Customizable alert systems

### 4. 📱 Android Application
**Location:** `android_app/`

- **Native Android development** with TensorFlow Lite
- On-device sensor data collection (accelerometer, gyroscope)
- Real-time activity classification
- Federated learning client implementation
- Live prediction streaming to private node

**Key Features:**
- Efficient on-device inference
- Background data collection
- Privacy-preserving local processing
- Seamless server communication
- User-friendly interface

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core development |
| pip | Latest | Package management |
| Android Studio | Latest | Mobile app development |


### Installation & Setup

#### 1. 📥 Clone Repository
```bash
git clone https://github.com/viki-777/momentum.git
cd momentum
```

#### 2. 🎯 Model Training
```bash
cd model_training/Scripts\ \(\ Offline\ Training\ \)/HAR_Momentum/
pip install -r requirements.txt
python training.py
```

#### 3. 🖥️ Start Federated Server
```bash
cd federated_server/src/server_deployment/
pip install flask waitress matplotlib numpy scikit-learn tensorflow
python server_with_metrics.py
```
📊 **Dashboard:** http://localhost:8080/dashboard

#### 4. 📡 Launch Private Node
```bash
cd federated_server/src/
python private_node.py
```
🔴 **Live Stream:** http://localhost:5000/

#### 5. 📱 Android App Setup
```bash
cd android_app/
# Open in Android Studio
# Build and deploy to device
# Configure server endpoints in app settings
```

## 📊 Usage Examples

### Training a New Model
```python
# Navigate to model_training/Scripts (Offline Training)/HAR_Momentum/
python training.py --dataset_path ./data --epochs 100 --batch_size 32
```

### Starting Federated Learning
```bash
# Start server
python server_with_metrics.py --port 8080 --clients 5

# Monitor progress at http://localhost:8080/dashboard
```

### Real-time Prediction Monitoring
```bash
# Start private node
python private_node.py --host 0.0.0.0 --port 5000

# View live predictions at http://localhost:5000/
```

## 📈 Evaluation & Simulation

The `evaluation_and_simulation/` directory contains:

- **Performance benchmarking** scripts
- **Federated learning simulation** environments  
- **Comparative analysis** tools
- **Custom evaluation metrics** for HAR tasks

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Add unit tests for new features
- Update documentation for API changes
- Ensure cross-platform compatibility

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **TensorFlow** team for ML framework
- **Flask** community for web framework
- **Android** development team
- **Federated Learning** research community
- **Scikit-learn** contributors
- Open source community



---

<div align="center">
  <p><strong>Built with ❤️ for the research community</strong></p>
  <p>Star ⭐ this repository if you find it helpful!</p>
</div>