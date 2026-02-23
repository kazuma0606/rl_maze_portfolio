# Design Document

## Overview

This document describes the design for a comprehensive reinforcement learning algorithm comparison system. The system will provide educational analysis comparing Random Agent, DQN (Deep Q-Network), and PPO (Proximal Policy Optimization) algorithms through an interactive Jupyter notebook.

The comparison will include:
- Performance metrics (success rate, rewards, training time)
- Neural network architecture visualization
- Algorithmic differences explanation
- Inference performance benchmarking
- Educational visualizations and explanations

## Architecture

### System Components

1. **Comparison Notebook** (`EX_algorithm_comparison.ipynb`)
   - Main entry point for all comparisons
   - Orchestrates model loading, evaluation, and visualization
   - Generates comparative analysis

2. **Model Loaders**
   - DQN model loader from `ml/models/dqn_*.zip`
   - PPO model loader from `ml/models/ppo_*.zip`
   - Random agent generator (no model required)

3. **Evaluation Engine**
   - Performance metrics calculator
   - Success rate evaluator
   - Reward aggregator
   - Training curve analyzer

4. **Visualization Module**
   - Architecture diagram generator (using torchviz/torchsummary)
   - Performance comparison charts
   - Learning curve comparisons
   - Inference speed benchmarks

## Components and Interfaces

### 1. Model Loading Interface

```python
def load_trained_model(algorithm: str, env_name: str) -> BaseAlgorithm:
    """
    Load a trained RL model
    
    Args:
        algorithm: 'dqn' or 'ppo'
        env_name: Environment name (e.g., 'cartpole', 'maze')
    
    Returns:
        Loaded model instance
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
```

### 2. Random Agent Interface

```python
class RandomAgent:
    """Simple random action agent for baseline comparison"""
    
    def predict(self, observation, deterministic=True):
        """Return random action"""
        return self.action_space.sample(), None
```

### 3. Evaluation Interface

```python
def evaluate_agent(agent, env, n_episodes: int = 100) -> Dict:
    """
    Evaluate agent performance
    
    Returns:
        {
            'mean_reward': float,
            'std_reward': float,
            'success_rate': float,
            'mean_episode_length': float
        }
    """
```

### 4. Architecture Visualization Interface

```python
def visualize_model_architecture(model, input_shape) -> None:
    """
    Generate and display model architecture diagram
    
    Uses torchviz or torchsummary to create visual representation
    """
```

## Data Models

### Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    algorithm_name: str
    mean_reward: float
    std_reward: float
    success_rate: float
    mean_episode_length: float
    inference_time_ms: float
    parameter_count: int
```

### Model Architecture Info

```python
@dataclass
class ModelArchitecture:
    algorithm_name: str
    network_type: str  # 'Q-Network', 'Actor', 'Critic'
    layers: List[LayerInfo]
    total_parameters: int
    trainable_parameters: int
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Acceptance Criteria Testing Prework

1.1 WHEN the comparison notebook executes THEN the Comparison System SHALL train or load all three agents (Random, DQN, PPO) on the same maze environment
Thoughts: This is about ensuring all three agents are properly initialized and ready for comparison. We can test this by verifying that each agent object exists and is properly configured for the same environment.
Testable: yes - example

1.2 WHEN evaluation completes THEN the Comparison System SHALL display success rates for all three agents with statistical confidence
Thoughts: This is about the output format and statistical validity. We can test that success rates are calculated correctly and confidence intervals are provided.
Testable: yes - property

1.3 WHEN evaluation completes THEN the Comparison System SHALL display average rewards for all three agents across multiple episodes
Thoughts: This is about aggregating rewards correctly across episodes for all agents.
Testable: yes - property

1.4 WHEN displaying results THEN the Comparison System SHALL present comparative visualizations showing performance differences
Thoughts: This is about UI/visualization quality, which is subjective.
Testable: no

1.5 WHEN training data exists THEN the Comparison System SHALL plot learning curves comparing DQN and PPO training progress
Thoughts: This is about correctly reading and plotting historical training data.
Testable: yes - example

2.1 WHEN the notebook requests architecture visualization THEN the Comparison System SHALL extract the PyTorch model structure from trained DQN agent
Thoughts: This is about successfully extracting model information from a specific agent type.
Testable: yes - example

2.2 WHEN the notebook requests architecture visualization THEN the Comparison System SHALL extract the PyTorch model structure from trained PPO agent
Thoughts: Similar to 2.1, but for PPO.
Testable: yes - example

2.3 WHEN displaying DQN architecture THEN the Comparison System SHALL show the Q-network layers, activation functions, and parameter counts
Thoughts: This is about the completeness of architecture information display.
Testable: yes - property

2.4 WHEN displaying PPO architecture THEN the Comparison System SHALL show both Actor and Critic networks with their respective structures
Thoughts: This is about correctly identifying and displaying both networks in PPO.
Testable: yes - property

2.5 WHEN comparing architectures THEN the Comparison System SHALL present side-by-side comparison highlighting structural differences
Thoughts: This is about visualization layout, which is subjective.
Testable: no

3.1-3.5: These are all about textual explanations and recommendations
Thoughts: These are educational content requirements, not functional requirements that can be tested programmatically.
Testable: no

4.1 WHEN measuring inference performance THEN the Comparison System SHALL record average inference time per action for each agent
Thoughts: This is about correctly measuring and recording timing data across all agents.
Testable: yes - property

4.2 WHEN measuring inference performance THEN the Comparison System SHALL execute at least 1000 inference calls to ensure statistical validity
Thoughts: This is a specific requirement about sample size.
Testable: yes - example

4.3 WHEN displaying performance metrics THEN the Comparison System SHALL show inference time comparison across all three agents
Thoughts: This is about displaying collected data.
Testable: yes - example

4.4 WHEN measuring model size THEN the Comparison System SHALL report total parameter counts for DQN and PPO models
Thoughts: This is about correctly counting model parameters.
Testable: yes - property

4.5 WHEN comparing resource usage THEN the Comparison System SHALL present memory footprint estimates for each model
Thoughts: This is about estimating memory usage.
Testable: yes - property

5.1-5.5: These are all about educational content, formatting, and presentation
Thoughts: These are UI/UX requirements that are subjective.
Testable: no

6.1 WHEN trained models exist in the ml/models directory THEN the Comparison System SHALL load existing DQN and PPO models
Thoughts: This is about file system operations and model loading.
Testable: yes - example

6.2 WHEN models are missing THEN the Comparison System SHALL provide clear instructions for training them
Thoughts: This is about error messaging.
Testable: yes - example

6.3 WHEN loading models THEN the Comparison System SHALL verify model compatibility with the current environment
Thoughts: This is about validation logic.
Testable: yes - property

6.4 WHEN models fail to load THEN the Comparison System SHALL display informative error messages with resolution steps
Thoughts: This is about error handling.
Testable: yes - example

6.5 WHEN using loaded models THEN the Comparison System SHALL indicate model source and training parameters in the output
Thoughts: This is about metadata display.
Testable: yes - example

### Property Reflection

After reviewing all testable properties:

**Redundancies identified:**
- Properties 1.2 and 1.3 both test evaluation output correctness - can be combined
- Properties 2.3 and 2.4 both test architecture information completeness - can be combined
- Properties 4.1, 4.4, and 4.5 all test metric collection - can be combined

**Consolidated properties:**

Property 1: Evaluation metrics completeness
*For any* set of agents (Random, DQN, PPO), evaluation should return complete metrics including success rate, mean reward, and episode length for each agent
**Validates: Requirements 1.2, 1.3**

Property 2: Architecture information completeness
*For any* trained model (DQN or PPO), architecture extraction should return all layers, activation functions, parameter counts, and network types (Q-Network for DQN, Actor/Critic for PPO)
**Validates: Requirements 2.3, 2.4**

Property 3: Performance metrics collection
*For any* agent, performance measurement should correctly record inference time, parameter count, and memory footprint
**Validates: Requirements 4.1, 4.4, 4.5**

Property 4: Model compatibility validation
*For any* loaded model, the system should verify that the model's observation and action spaces match the target environment
**Validates: Requirements 6.3**

## Error Handling

### Model Loading Errors

- **FileNotFoundError**: Model file doesn't exist
  - Action: Display clear message with path to expected model location
  - Provide instructions for training the model

- **VersionMismatchError**: Model was trained with incompatible library version
  - Action: Display version information
  - Suggest retraining or updating dependencies

- **EnvironmentMismatchError**: Model trained for different environment
  - Action: Display expected vs actual environment specs
  - Prevent evaluation with incompatible model

### Evaluation Errors

- **EnvironmentError**: Environment fails during evaluation
  - Action: Log error details
  - Continue with remaining agents if possible

- **TimeoutError**: Evaluation takes too long
  - Action: Set reasonable timeout limits
  - Display partial results if available

## Testing Strategy

### Unit Testing

Unit tests will cover:
- Model loading functions
- Metric calculation functions
- Architecture extraction functions
- Random agent behavior

Example unit tests:
```python
def test_load_dqn_model():
    """Test DQN model loading"""
    model = load_trained_model('dqn', 'cartpole')
    assert model is not None
    assert hasattr(model, 'predict')

def test_random_agent_actions():
    """Test random agent generates valid actions"""
    env = gym.make('CartPole-v1')
    agent = RandomAgent(env.action_space)
    action, _ = agent.predict(None)
    assert env.action_space.contains(action)
```

### Property-Based Testing

Property-based tests will use **Hypothesis** (Python's property-based testing library) to verify universal properties.

The model MUST configure each property-based test to run a minimum of 100 iterations.

Each property-based test MUST be tagged with a comment explicitly referencing the correctness property in the design document using this exact format: '**Feature: rl-algorithm-comparison, Property {number}: {property_text}**'

Property tests will cover:

**Property 1: Evaluation metrics completeness**
```python
@given(agent_type=st.sampled_from(['random', 'dqn', 'ppo']))
@settings(max_examples=100)
def test_evaluation_completeness(agent_type):
    """
    **Feature: rl-algorithm-comparison, Property 1: Evaluation metrics completeness**
    
    For any agent type, evaluation should return complete metrics
    """
    agent = create_agent(agent_type)
    metrics = evaluate_agent(agent, env, n_episodes=10)
    
    assert 'mean_reward' in metrics
    assert 'std_reward' in metrics
    assert 'success_rate' in metrics
    assert 'mean_episode_length' in metrics
    assert all(isinstance(v, (int, float)) for v in metrics.values())
```

**Property 2: Architecture information completeness**
```python
@given(model_type=st.sampled_from(['dqn', 'ppo']))
@settings(max_examples=100)
def test_architecture_extraction(model_type):
    """
    **Feature: rl-algorithm-comparison, Property 2: Architecture information completeness**
    
    For any trained model, architecture extraction should return complete information
    """
    model = load_trained_model(model_type, 'cartpole')
    arch_info = extract_architecture(model)
    
    assert 'layers' in arch_info
    assert 'parameter_count' in arch_info
    assert len(arch_info['layers']) > 0
    
    if model_type == 'ppo':
        assert 'actor' in arch_info
        assert 'critic' in arch_info
```

**Property 3: Performance metrics collection**
```python
@given(agent_type=st.sampled_from(['random', 'dqn', 'ppo']))
@settings(max_examples=100)
def test_performance_metrics(agent_type):
    """
    **Feature: rl-algorithm-comparison, Property 3: Performance metrics collection**
    
    For any agent, performance measurement should correctly record all metrics
    """
    agent = create_agent(agent_type)
    perf = measure_performance(agent, env, n_calls=1000)
    
    assert 'inference_time_ms' in perf
    assert perf['inference_time_ms'] > 0
    
    if agent_type != 'random':
        assert 'parameter_count' in perf
        assert perf['parameter_count'] > 0
```

**Property 4: Model compatibility validation**
```python
@given(
    model_env=st.sampled_from(['cartpole', 'maze']),
    target_env=st.sampled_from(['cartpole', 'maze'])
)
@settings(max_examples=100)
def test_model_compatibility(model_env, target_env):
    """
    **Feature: rl-algorithm-comparison, Property 4: Model compatibility validation**
    
    For any loaded model, compatibility check should correctly identify mismatches
    """
    model = load_trained_model('dqn', model_env)
    env = create_environment(target_env)
    
    is_compatible = check_compatibility(model, env)
    
    if model_env == target_env:
        assert is_compatible
    else:
        assert not is_compatible
```

### Integration Testing

Integration tests will verify:
- End-to-end notebook execution
- Correct interaction between components
- Visualization generation
- Complete comparison workflow

### Testing Requirements Summary

- Unit tests verify specific examples and edge cases
- Property-based tests verify universal properties across all inputs
- Both types of tests are complementary and essential
- Property tests use Hypothesis with minimum 100 iterations
- Each property test explicitly references design document properties
- Tests focus on functional correctness, not UI appearance

## Implementation Notes

### Dependencies

- `stable-baselines3`: For DQN and PPO models
- `gymnasium`: For environment interface
- `torch`: For model inspection
- `torchviz` or `torchsummary`: For architecture visualization
- `matplotlib`: For plotting
- `numpy`: For numerical operations
- `hypothesis`: For property-based testing

### Model File Locations

- DQN models: `ml/models/dqn_*.zip`
- PPO models: `ml/models/ppo_*.zip`
- Training logs: `ml/experiments/logs/`

### Performance Considerations

- Cache loaded models to avoid repeated file I/O
- Use vectorized operations for metric calculations
- Limit visualization complexity for large models
- Set reasonable timeouts for evaluation

## Future Enhancements

- Support for additional algorithms (A3C, SAC, TD3)
- Interactive parameter tuning
- Real-time training comparison
- Export comparison reports to PDF
- Integration with TensorBoard for detailed analysis
