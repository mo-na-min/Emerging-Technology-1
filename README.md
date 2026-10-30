# Real-Time Filipino Sign Language (FSL) Translator

This desktop application translates Filipino Sign Language (FSL) in real time using webcam input. It supports a two-tier recognition architecture combining static letters/numbers and dynamic word signs.

---

## 1. Installation & Environment Setup

Ensure Python 3.9+ is installed on your system.

```bash
# Clone or navigate to the repository directory
cd fsl_translator

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt