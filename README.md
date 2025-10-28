# Leatherback Vehicle RL Environment

A reinforcement learning environment for training autonomous vehicles using the NVIDIA Leatherback car in Isaac Lab. This project provides a waypoint-following task where an RL agent learns to navigate through a sequence of waypoints while maintaining appropriate heading alignment.

**Inspired by [Lychee AI's Leatherback project](https://github.com/MuammerBay/Leatherback/blob/main/README.md).**

## 🚗 Overview

The environment simulates a 4-wheel drive vehicle that must:
- Navigate through 10 randomly generated waypoints per episode
- Maintain heading toward target waypoints
- Learn autonomous driving using RL algorithms (PPO, MAPPO, etc.)

**Visualizations**: Red spheres = current target, Green = future waypoints, Cyan/Red arrows = heading markers

## 📋 Prerequisites

### Compute Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
  - Minimum: 6GB VRAM
  - Recommended: 8GB+ VRAM for 4096 parallel environments
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB+ free space for Isaac Sim

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11 (64-bit)
- **Isaac Sim**: 4.1.0 or later
- **Python**: 3.8 - 3.10
- **CUDA**: 11.8 or 12.1+ (must match Isaac Sim requirements)
- **Git**: For cloning the repository

## 📦 Installation

### Step 1: Install Isaac Lab

Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

**Recommended**: Use conda or uv installation for easier Python script execution.

### Step 2: Clone This Repository

Clone outside your `IsaacLab` directory:

```bash
git clone <your-repo-url>
cd leatherback
```

### Step 3: Install Extension

#### Linux:
```bash
# If Isaac Lab is in conda/venv
python -m pip install -e source/leatherback

# If using Isaac Lab launcher script
./isaaclab.sh -p python -m pip install -e source/leatherback
```

#### Windows:
```powershell
# If Isaac Lab is in conda/venv
python -m pip install -e source/leatherback

# If using Isaac Lab launcher script
isaaclab.bat -p python -m pip install -e source/leatherback
```

### Step 4: Verify Installation

```bash
# Linux/Windows (adjust for your setup)
python scripts/list_envs.py
```

You should see `Template-Leatherback-Direct-v0` in the output.

## 🚀 Running the Project

### Linux

**Training**:
```bash
# Basic training with skrl
./isaaclab.sh -p scripts/skrl/train.py --task=Template-Leatherback-Direct-v0

# With options
./isaaclab.sh -p scripts/skrl/train.py --task=Template-Leatherback-Direct-v0 --num_envs=2048 --seed=42 --video
```

**Testing with dummy agents**:
```bash
./isaaclab.sh -p scripts/zero_agent.py --task=Template-Leatherback-Direct-v0
./isaaclab.sh -p scripts/random_agent.py --task=Template-Leatherback-Direct-v0
```

**Inference**:
```bash
./isaaclab.sh -p scripts/skrl/play.py --task=Template-Leatherback-Direct-v0-Play --checkpoint=path/to/checkpoint.pth
```

### Windows

**Training**:
```powershell
# Basic training with skrl
isaaclab.bat -p scripts\skrl\train.py --task=Template-Leatherback-Direct-v0

# With options
isaaclab.bat -p scripts\skrl\train.py --task=Template-Leatherback-Direct-v0 --num_envs=2048 --seed=42 --video
```

**Testing with dummy agents**:
```powershell
isaaclab.bat -p scripts\zero_agent.py --task=Template-Leatherback-Direct-v0
isaaclab.bat -p scripts\random_agent.py --task=Template-Leatherback-Direct-v0
```

**Inference**:
```powershell
isaaclab.bat -p scripts\skrl\play.py --task=Template-Leatherback-Direct-v0-Play --checkpoint=path\to\checkpoint.pth
```

### Using Conda/Virtual Environment

If Isaac Lab is installed in a conda environment, activate it first:

```bash
# Activate environment
conda activate <isaaclab-env-name>  # or: source activate <env-name>

# Then run without launcher script
python scripts/skrl/train.py --task=Template-Leatherback-Direct-v0 --num_envs=2048
```

### Common Training Options

```bash
--task=Template-Leatherback-Direct-v0    # Task name
--num_envs=2048                          # Number of parallel environments (default: 4096)
--seed=42                                # Random seed
--video                                  # Record training videos
--checkpoint=path/to/checkpoint.pth      # Resume from checkpoint
--max_iterations=10000                   # Training iterations
```

## 🤝 Contributing

1. **Fork & Branch**: Create a feature branch from your fork
2. **Code Style**: Follow existing patterns, add comments for complex logic
3. **Test**: Verify with dummy agents and small `--num_envs`
4. **Format**: `pip install pre-commit && pre-commit run --all-files`
5. **Submit PR**: Clear description of changes

**Areas**: Reward shaping, new environments, multi-agent enhancements, bug fixes

## 📂 Code Structure

```
source/leatherback/leatherback/
├── robots/leatherback.py              # Vehicle config & actuators
└── tasks/direct/leatherback/
    ├── leatherback_env.py             # Environment implementation
    ├── leatherback_env_cfg.py         # Configuration
    ├── waypoints.py                   # Waypoint markers
    └── markers.py                     # Heading visualization
```

## 🛠️ Optional Setup

### IDE Setup (VS Code)
1. Press `Ctrl+Shift+P` → `Tasks: Run Task` → `setup_python_env`
2. Enter Isaac Sim installation path
3. Add to `.vscode/settings.json` if needed:
   ```json
   {
     "python.analysis.extraPaths": ["<path>/source/leatherback"]
   }
   ```

### Omniverse Extension
1. Isaac Sim → `Window` → `Extensions`
2. Hamburger icon → `Settings` → Add `source/` path to `Extension Search Paths`
3. Refresh → Enable under `Third Party`

## 📚 Resources

- [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Lychee AI Leatherback](https://github.com/MuammerBay/Leatherback)

## 🔧 Troubleshooting

**Joint position errors**: Ensure shock joint positions are within valid ranges (see `leatherback.py`)

**Pylance issues**: Add extension path to `.vscode/settings.json` under `python.analysis.extraPaths`

**Out of memory**: Reduce `--num_envs` or upgrade GPU/VRAM

## 📄 License

BSD-3-Clause (same as Isaac Lab)
