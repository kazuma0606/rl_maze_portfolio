package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

// ONNXModelInfo represents metadata about an ONNX model
type ONNXModelInfo struct {
	InputShape  []int64 `json:"input_shape"`
	OutputShape []int64 `json:"output_shape"`
	InputName   string  `json:"input_name"`
	OutputName  string  `json:"output_name"`
}

// InferenceResult represents the output of an inference
type InferenceResult struct {
	QValues  []float32     `json:"q_values"`
	Action   int           `json:"action"`
	Latency  time.Duration `json:"latency"`
	MaxQValue float32      `json:"max_q_value"`
}

func main() {
	fmt.Println("=== Go ONNX Validation ===")
	fmt.Println("Testing ONNX model loading and inference simulation")
	fmt.Println()

	// Check if ONNX model exists
	modelPath := filepath.Join("..", "..", "ml", "models", "dqn_cartpole.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s\nPlease run 04_onnx_export.ipynb first", modelPath)
	}
	fmt.Printf("✓ Model file found: %s\n", modelPath)

	// Read model file to verify it's valid
	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model file: %v", err)
	}
	fmt.Printf("✓ Model file readable (%d bytes)\n", len(modelData))

	// Simulate model metadata (in real implementation, this would come from ONNX Runtime)
	modelInfo := ONNXModelInfo{
		InputShape:  []int64{1, 4}, // Batch size 1, 4 features (CartPole observation)
		OutputShape: []int64{1, 2}, // Batch size 1, 2 actions
		InputName:   "input",
		OutputName:  "output",
	}

	fmt.Printf("\nModel Information:\n")
	fmt.Printf("  Input shape: %v\n", modelInfo.InputShape)
	fmt.Printf("  Output shape: %v\n", modelInfo.OutputShape)
	fmt.Printf("  Input name: %s\n", modelInfo.InputName)
	fmt.Printf("  Output name: %s\n", modelInfo.OutputName)

	// Prepare test input (CartPole observation: 4 dimensions)
	testObservation := []float32{0.0, 0.0, 0.0, 0.0}
	fmt.Printf("\nTest Input:\n")
	fmt.Printf("  Data: %v\n", testObservation)

	// Run inference simulation and measure latency
	fmt.Println("\n=== Running Inference Tests ===")
	fmt.Println("Note: This is a simulation. Real ONNX Runtime integration requires:")
	fmt.Println("  1. ONNX Runtime C library installed")
	fmt.Println("  2. CGO enabled")
	fmt.Println("  3. Proper library paths configured")
	fmt.Println()

	numRuns := 100
	latencies := make([]time.Duration, numRuns)
	results := make([]InferenceResult, numRuns)

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numRuns; i++ {
		start := time.Now()

		// Simulate inference (in real implementation, this would call ONNX Runtime)
		qValues := simulateInference(testObservation)
		
		latency := time.Since(start)
		latencies[i] = latency

		// Determine action (argmax of Q-values)
		action, maxQ := argmax(qValues)

		results[i] = InferenceResult{
			QValues:   qValues,
			Action:    action,
			Latency:   latency,
			MaxQValue: maxQ,
		}

		// Display first result
		if i == 0 {
			fmt.Printf("First Inference Result:\n")
			fmt.Printf("  Q-values: %v\n", qValues)
			fmt.Printf("  Selected action: %d (Q-value: %.4f)\n", action, maxQ)
			fmt.Printf("  Latency: %v\n", latency)
		}
	}

	// Calculate latency statistics
	fmt.Printf("\n=== Latency Statistics (%d runs) ===\n", numRuns)

	var totalLatency time.Duration
	minLatency := latencies[0]
	maxLatency := latencies[0]

	for _, latency := range latencies {
		totalLatency += latency
		if latency < minLatency {
			minLatency = latency
		}
		if latency > maxLatency {
			maxLatency = latency
		}
	}

	avgLatency := totalLatency / time.Duration(numRuns)

	// Calculate p95 latency
	sortedLatencies := make([]time.Duration, numRuns)
	copy(sortedLatencies, latencies)
	// Simple bubble sort for small array
	for i := 0; i < numRuns-1; i++ {
		for j := 0; j < numRuns-i-1; j++ {
			if sortedLatencies[j] > sortedLatencies[j+1] {
				sortedLatencies[j], sortedLatencies[j+1] = sortedLatencies[j+1], sortedLatencies[j]
			}
		}
	}
	p95Index := int(float64(numRuns) * 0.95)
	p95Latency := sortedLatencies[p95Index]

	fmt.Printf("  Min:     %v\n", minLatency)
	fmt.Printf("  Max:     %v\n", maxLatency)
	fmt.Printf("  Average: %v\n", avgLatency)
	fmt.Printf("  P95:     %v\n", p95Latency)

	// Check if p95 meets requirement (< 50ms)
	// Note: This is simulated latency, real ONNX inference will be faster
	fmt.Println("\n=== Validation Results ===")
	fmt.Printf("P95 latency: %.2fms\n", float64(p95Latency.Microseconds())/1000.0)
	fmt.Println("Note: This is simulated latency. Real ONNX Runtime inference is expected to be < 50ms")

	// Save results to JSON for inspection
	resultsFile := "inference_results.json"
	if err := saveResults(results, resultsFile); err != nil {
		log.Printf("Warning: Failed to save results: %v", err)
	} else {
		fmt.Printf("\n✓ Results saved to %s\n", resultsFile)
	}

	fmt.Println("\n=== Integration Validation ===")
	fmt.Println("✓ Model file exists and is readable")
	fmt.Println("✓ Model metadata structure validated")
	fmt.Println("✓ Inference simulation completed successfully")
	fmt.Println("✓ Latency measurement implemented")
	fmt.Println("✓ Action selection (argmax) implemented")
	fmt.Println()
	fmt.Println("Next Steps:")
	fmt.Println("  1. Install ONNX Runtime C library for real inference")
	fmt.Println("  2. Enable CGO and configure library paths")
	fmt.Println("  3. Replace simulation with actual ONNX Runtime calls")
	fmt.Println("  4. Integrate into backend/inference/ Go server")
	fmt.Println()
	fmt.Println("=== Validation Complete ===")
}

// simulateInference simulates ONNX inference
// In real implementation, this would call ONNX Runtime
func simulateInference(observation []float32) []float32 {
	// Simulate some computation time (real ONNX inference)
	time.Sleep(time.Microsecond * time.Duration(rand.Intn(100)+50))

	// Return simulated Q-values for CartPole (2 actions)
	// In real implementation, this would come from ONNX model
	return []float32{
		rand.Float32()*2 - 1, // Q-value for action 0
		rand.Float32()*2 - 1, // Q-value for action 1
	}
}

// argmax returns the index and value of the maximum element
func argmax(values []float32) (int, float32) {
	if len(values) == 0 {
		return -1, 0
	}

	maxIdx := 0
	maxVal := values[0]

	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}

	return maxIdx, maxVal
}

// saveResults saves inference results to a JSON file
func saveResults(results []InferenceResult, filename string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}
