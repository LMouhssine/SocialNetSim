# SocialNetSim

**AI-Powered Synthetic Social Network and Virality Simulator**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

**SocialNetSim** is a comprehensive simulation framework for studying social network dynamics, viral content propagation, and misinformation spread. Built entirely from synthetic data, it enables researchers and developers to:

- Model realistic social networks with configurable user traits and behaviors
- Simulate content virality and engagement patterns
- Test content moderation strategies and feed algorithms
- Train AI models for virality prediction and misinformation detection
- Run what-if experiments without privacy concerns or data collection

**All data is synthetically generated** — no external datasets required.

---

## Motivation & Use Cases

### Research Applications
- **Misinformation Spread**: Study how false information propagates through networks under different conditions
- **Echo Chambers**: Analyze the formation and effects of ideological filter bubbles
- **Platform Design**: Test feed algorithms, moderation policies, and network effects before deployment
- **Viral Dynamics**: Understand what makes content go viral and how cascades form

### Educational Use
- Teaching social network analysis without privacy concerns
- Demonstrating machine learning in social media contexts
- Exploring computational social science concepts hands-on

### Industry Applications
- A/B testing platform features in a controlled environment
- Training content moderation systems
- Developing recommendation algorithms
- Prototyping social features before production

---

## Key Features

### 🌐 Synthetic World Generation
- **Realistic Networks**: Enhanced Barabási-Albert model with interest and ideology-based attachment
- **Diverse Users**: Configurable traits including ideology, confirmation bias, emotional reactivity, and misinformation susceptibility
- **Rich Content**: Auto-generated posts with topics, sentiment, emotions, and quality scores

### 📊 Advanced Simulation Engine
- **Multi-Algorithm Feed Ranking**: Chronological, engagement-based, diverse, and interest-based feeds
- **Engagement Modeling**: Content-user matching, social influence, and temporal dynamics
- **Viral Cascades**: Threshold-based sharing models with exponential spread
- **Random Events**: Political shocks, misinformation waves, viral trends, and algorithm changes
- **Content Moderation**: Configurable detection accuracy and suppression strategies

### 🤖 Built-in AI Models
- **Virality Predictor**: XGBoost-based model for predicting viral content using early signals
- **Churn Predictor**: Forecast user inactivity and platform abandonment
- **Misinformation Detector**: Identify potentially false content based on features and patterns

### 🧪 Experiment Framework
- **What-If Analysis**: Compare different configurations side-by-side
- **Pre-built Scenarios**: Echo chamber, misinformation, and balanced configurations
- **Reproducible Results**: Seed-based reproducibility for scientific rigor

### 🎨 Interactive Dashboard
- **Streamlit UI**: User-friendly interface for configuration and visualization
- **Network Visualization**: Interactive network graphs with pyvis
- **Real-time Metrics**: Live charts tracking engagement, cascades, and polarization
- **Model Training**: Train and evaluate AI models directly from the UI

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LMouhssine/SocialNetSim.git
cd SocialNetSim

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to access the interactive dashboard.

### Command Line Usage

```bash
# Generate a synthetic world
python scripts/generate_world.py --users 1000 --seed 42

# Run a simulation
python scripts/run_simulation.py --users 1000 --steps 100 --output data/simulations/my_run

# Train AI models
python scripts/train_models.py --simulation data/simulations/my_run --output data/models
```

### Docker

```bash
docker-compose -f docker/docker-compose.yaml up --build
```

Access the dashboard at [http://localhost:8501](http://localhost:8501)

---

## Architecture

### Project Structure

```
SocialNetSim/
├── config/              # Configuration schemas and YAML files
│   ├── schemas.py      # Pydantic models for validation
│   ├── base.yaml       # Default configuration
│   └── scenarios/      # Pre-built scenario configs
├── models/              # Core data models
│   ├── user.py         # User and UserTraits
│   ├── post.py         # Post and PostContent
│   ├── interaction.py  # Interaction and Cascade
│   └── event.py        # Event and EventEffect
├── generator/           # Synthetic data generation
│   ├── world.py        # World orchestrator
│   ├── user_generator.py
│   ├── network_generator.py
│   ├── content_generator.py
│   └── topic_generator.py
├── engine/              # Simulation engine
│   ├── simulation.py   # Main simulation loop
│   ├── state.py        # State management
│   ├── feed.py         # Feed ranking algorithms
│   ├── engagement.py   # Engagement probability model
│   ├── cascade.py      # Viral cascade spreading
│   ├── events.py       # Random event generation
│   └── moderation.py   # Content moderation
├── ai/                  # Machine learning models
│   ├── features.py     # Feature engineering
│   ├── evaluation.py   # Model evaluation
│   └── trainers/       # ML model trainers
├── scenarios/           # Experiment framework
│   ├── experiment.py   # Experiment runner
│   ├── comparator.py   # Result comparison
│   └── presets.py      # Pre-built experiments
├── ui/                  # Streamlit dashboard
│   ├── app.py          # Main entry point
│   ├── pages/          # Dashboard pages
│   └── components/     # Reusable UI components
├── scripts/             # CLI tools
├── tests/               # Test suite
└── docker/              # Docker configuration
```

### Core Components

#### Network Generation
Enhanced Barabási-Albert model where attachment probability combines:
```
P = w_degree * (degree_i / total_degree) +
    w_interest * cosine_similarity(interests) +
    w_ideology * (1 - |ideology_diff| / 2)
```

Default weights: degree (0.5), interest (0.3), ideology (0.2)

#### Engagement Model
User engagement probability calculated as:
```
P(engage) = base_rate × content_match × quality_factor × social_factor × temporal_factor
```

**Factors:**
- **content_match**: Topic interest + ideology alignment (weighted by confirmation bias)
- **quality_factor**: Content quality amplified by emotional reactivity
- **social_factor**: Author influence + friend engagement count
- **temporal_factor**: Post freshness decay + user fatigue

#### Cascade Spreading
Content spreads through the network using:
- **Exposure probability**: Base rate × velocity boost × virality potential × decay
- **Threshold model**: Users share if ≥ N friends already shared

---

## Configuration

### Pre-built Scenarios

| Scenario | Description | Use Case |
|----------|-------------|----------|
| `default` | Balanced configuration | General exploration and baseline |
| `echo_chamber` | High confirmation bias, ideology-driven connections | Study filter bubbles and polarization |
| `misinformation` | Elevated misinfo rates, varied susceptibility | Analyze false information spread |

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `user.num_users` | Number of synthetic users | 1000 |
| `network.weight_ideology` | Ideology influence on connections | 0.2 |
| `content.misinformation_rate` | Base rate of misinformation | 0.05 |
| `feed.algorithm` | Feed ranking algorithm | engagement |
| `moderation.enabled` | Enable content moderation | true |
| `moderation.detection_accuracy` | Accuracy of misinfo detection | 0.8 |

### Example Configuration

```python
from config.schemas import SimulationConfig

config = SimulationConfig(
    name="my_simulation",
    seed=42,
    num_steps=100
)

# Customize parameters
config.user.num_users = 1000
config.network.weight_ideology = 0.5  # Stronger ideological clustering
config.content.misinformation_rate = 0.1  # Higher misinfo rate
config.feed.algorithm = "diverse"  # Use diversity-promoting feed

# Load a scenario
from config.schemas import load_scenario
config = load_scenario("echo_chamber")
```

---

## Running Experiments

### Algorithm Comparison

```python
from scenarios import create_algorithm_comparison

# Compare feed algorithms
experiment = create_algorithm_comparison(
    num_users=500,
    num_steps=100,
    num_runs=3
)

results = experiment.run()
print(results.comparison_summary)
```

### Custom Experiment

```python
from scenarios import Experiment, ExperimentConfig
from config.schemas import SimulationConfig

config = ExperimentConfig(
    name="Moderation Impact Study",
    base_config=SimulationConfig(seed=42),
    variations={
        "no_moderation": {
            "moderation": {"enabled": False}
        },
        "light_moderation": {
            "moderation": {"detection_accuracy": 0.5}
        },
        "aggressive_moderation": {
            "moderation": {"detection_accuracy": 0.95}
        }
    },
    num_runs=5
)

experiment = Experiment(config)
results = experiment.run()
```

---

## AI Models

### Virality Predictor

Predicts whether posts will go viral based on:
- **Content features**: Quality, controversy, emotions, sentiment
- **Author features**: Influence, credibility, follower count
- **Early signals**: Velocity, like rate, share rate in first N steps

```python
from ai.trainers import ViralityPredictor

predictor = ViralityPredictor(viral_threshold=50)
metrics = predictor.train_from_simulation(posts, users, state)

# Make predictions
is_viral, probability = predictor.predict_virality(new_post, author, state)
```

### Churn Predictor

Forecasts user inactivity based on:
- User traits and engagement history
- Network position and influence
- Recent activity patterns

### Misinformation Detector

Identifies potential misinformation using:
- Content signals (quality, emotional intensity, controversy)
- Author credibility scores
- Engagement patterns and velocity

---

## API Reference

### Basic Usage

```python
from config.schemas import SimulationConfig
from generator import World
from engine import Simulation

# Create configuration
config = SimulationConfig(num_steps=100, seed=42)
config.user.num_users = 1000

# Build synthetic world
world = World(config)
world.build()

# Run simulation
sim = Simulation(config)
sim.initialize(world)
results = sim.run()

# Analyze results
metrics_df = sim.get_metrics_dataframe()
print(results["metrics_summary"])
```

### Advanced: Custom Callbacks

```python
def my_callback(step, metrics):
    print(f"Step {step}: {metrics.new_posts} posts, {metrics.new_interactions} interactions")

sim.add_step_callback(my_callback)
sim.run()
```

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_engine/test_simulation.py -v
```

---

## Roadmap

- [ ] Real-time simulation stepping in UI
- [ ] Network community detection algorithms
- [ ] Recommendation system simulation
- [ ] Bot and coordinated behavior injection
- [ ] Multi-platform simulation (cross-network dynamics)
- [ ] REST API for integration with other tools
- [ ] Advanced visualization (animated network evolution)
- [ ] Export results to standard formats (GraphML, JSON-LD)

---

## Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository** and create a feature branch
2. **Add tests** for new functionality
3. **Follow code style** (we use Black for formatting)
4. **Write clear commit messages**
5. **Submit a pull request** with a description of your changes

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest

# Format code
black .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use SocialNetSim in your research, please cite:

```bibtex
@software{socialnetsimd2025,
  title = {SocialNetSim: AI-Powered Synthetic Social Network Simulator},
  author = {Lakhili, Mouhssine},
  year = {2025},
  url = {https://github.com/LMouhssine/SocialNetSim}
}
```

---

## Acknowledgments

Built with:
- [NetworkX](https://networkx.org/) for graph operations
- [XGBoost](https://xgboost.readthedocs.io/) for machine learning
- [Streamlit](https://streamlit.io/) for the interactive dashboard
- [Pydantic](https://pydantic.dev/) for configuration validation
- [Plotly](https://plotly.com/) for visualizations

---

## Support

- **Issues**: [GitHub Issues](https://github.com/LMouhssine/SocialNetSim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LMouhssine/SocialNetSim/discussions)

---

**Star this repo** if you find it useful! ⭐
