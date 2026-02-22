# Go ONNX Validation

This directory contains a Go program to validate ONNX model loading and inference patterns.

## Purpose

- Validate ONNX model file structure
- Demonstrate inference patterns in Go
- Measure inference latency
- Validate integration approach for the main inference server

## Implementation

This validation uses a **simulation approach** to demonstrate the ONNX integration pattern without requiring ONNX Runtime C library installation. The simulation:

- Loads and validates the ONNX model file
- Simulates inference with realistic latency
- Implements action selection (argmax)
- Measures and reports latency statistics
- Saves results to JSON for inspection

For **real ONNX Runtime integration**, see `main_with_onnx.go.example` which demonstrates the actual ONNX Runtime API usage.

## Prerequisites

- **Go 1.21+**: Ensure Go is installed: `go version`
- **ONNX Model**: The DQN CartPole model should exist at `../../ml/models/dqn_cartpole.onnx`
  - Run `04_onnx_export.ipynb` to generate the model if needed

## Setup

1. Navigate to the directory:
   ```bash
   cd experiments/go_onnx_validation
   ```

2. Download dependencies:
   ```bash
   go mod download
   ```

## Running

```bash
go run main.go
```

Or build and run:
```bash
go build -o validation.exe main.go
./validation.exe
```

## Expected Output

The program will:
1. Validate the ONNX model file exists and is readable
2. Display model metadata (input/output shapes)
3. Run 100 inference simulations
4. Display the first inference result (Q-values, selected action)
5. Calculate and display latency statistics (min, max, average, p95)
6. Save all results to `inference_results.json`
7. Display validation checklist

Example output:
```
=== Go ONNX Validation ===
Testing ONNX model loading and inference simulation

✓ Model file found: ..\..\ml\models\dqn_cartpole.onnx
✓ Model file readable (12932 bytes)

Model Information:
  Input shape: [1 4]
  Output shape: [1 2]
  Input name: input
  Output name: output

Test Input:
  Data: [0 0 0 0]

=== Running Inference Tests ===
Note: This is a simulation. Real ONNX Runtime integration requires:
  1. ONNX Runtime C library installed
  2. CGO enabled
  3. Proper library paths configured

First Inference Result:
  Q-values: [-0.5645006 0.5948441]
  Selected action: 1 (Q-value: 0.5948)
  Latency: 578.3µs

=== Latency Statistics (100 runs) ===
  Min:     505.6µs
  Max:     602.5µs
  Average: 520.351µs
  P95:     578µs

=== Validation Results ===
P95 latency: 0.58ms
Note: This is simulated latency. Real ONNX Runtime inference is expected to be < 50ms

✓ Results saved to inference_results.json

=== Integration Validation ===
✓ Model file exists and is readable
✓ Model metadata structure validated
✓ Inference simulation completed successfully
✓ Latency measurement implemented
✓ Action selection (argmax) implemented

Next Steps:
  1. Install ONNX Runtime C library for real inference
  2. Enable CGO and configure library paths
  3. Replace simulation with actual ONNX Runtime calls
  4. Integrate into backend/inference/ Go server

=== Validation Complete ===
```

## Real ONNX Runtime Integration

To use real ONNX Runtime (optional):

1. **Install ONNX Runtime C Library**

   **Windows:**
   - Download from https://github.com/microsoft/onnxruntime/releases
   - Extract and add the `lib` directory to your PATH
   - Set: `set CGO_LDFLAGS=-L"C:\path\to\onnxruntime\lib"`

   **Linux:**
   ```bash
   sudo apt-get install libonnxruntime-dev
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```

   **macOS:**
   ```bash
   brew install onnxruntime
   export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
   ```

2. **Use the Real Implementation**
   ```bash
   # Rename the example file
   mv main_with_onnx.go.example main_with_onnx.go
   
   # Update go.mod to include ONNX Runtime
   # Add: require github.com/yalue/onnxruntime_go v1.11.0
   
   # Run
   go mod tidy
   go run main_with_onnx.go
   ```

## Files

- `main.go` - Simulation-based validation (no external dependencies)
- `main_with_onnx.go.example` - Real ONNX Runtime integration example
- `go.mod` - Go module definition
- `README.md` - This file
- `inference_results.json` - Generated results (gitignored)

## Integration with Main System

This validation demonstrates the patterns that will be used in `backend/inference/`:

1. **Model Loading**: Reading ONNX files and validating structure
2. **Inference Execution**: Running predictions on observations
3. **Latency Measurement**: Tracking performance metrics
4. **Action Selection**: Implementing argmax for Q-value selection
5. **Result Serialization**: Saving results in JSON format

The main inference server will use the same approach with real ONNX Runtime integration.

## Validation Checklist

- [x] Model file validation
- [x] Model metadata extraction
- [x] Inference pattern implementation
- [x] Latency measurement
- [x] Action selection (argmax)
- [x] Result serialization
- [x] Error handling
- [ ] Real ONNX Runtime integration (optional, requires C library)
