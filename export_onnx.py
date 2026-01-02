"""
ONNX Export Module for XAUUSD Trading System

This module converts trained Stable Baselines3 models to ONNX format
for integration with MetaTrader5 via MQL5's OnnxRun function.

Author: AI Trading System
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import ONNX libraries (optional - only needed for export)
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    print(f"Warning: ONNX libraries not available: {e}")
    print("ONNX export functionality will be disabled.")
    print("To enable: pip install onnx onnxruntime")
except Exception as e:
    ONNX_AVAILABLE = False
    print(f"Warning: ONNX libraries failed to load: {e}")
    print("This may be due to missing DLL dependencies or Python version incompatibility.")
    print("ONNX export functionality will be disabled.")


def export_ppo_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: Tuple[int, ...] = (362,),  # Updated for candlestick sequence: 50*7 + 6 + 6
    input_name: str = "observations",
    output_name: str = "actions",
    opset_version: int = 11,
    verbose: bool = True
) -> str:
    """
    Export a Stable Baselines3 PPO model to ONNX format.
    
    Args:
        model_path: Path to the saved SB3 model
        output_path: Path to save the ONNX model
        input_shape: Shape of input observations (default: (362,) for candlestick sequence)
        input_name: Name of input layer (must match MQL5 expectations)
        output_name: Name of output layer (must match MQL5 expectations)
        opset_version: ONNX opset version (default: 11 for compatibility)
        verbose: Whether to print verbose information
        
    Returns:
        Path to the saved ONNX model
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError(
            "ONNX libraries are not available. "
            "Please install them with: pip install onnx onnxruntime\n"
            "Or if you encounter DLL errors, try: pip install onnxruntime"
        )
    
    if verbose:
        print(f"Loading model from {model_path}...")
    
    # Load the SB3 model
    model = PPO.load(model_path)
    
    # Get the policy network (actor network for action prediction)
    policy = model.policy
    
    # Create a dummy input for tracing
    dummy_input = torch.randn(1, *input_shape)
    
    if verbose:
        print(f"Converting policy network to ONNX...")
        print(f"Input shape: {input_shape}")
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
    
    # Export the actor network (policy) to ONNX
    # We only need the actor network for inference, not the critic
    try:
        # Create a wrapper that extracts the actor network
        class ActorWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
            
            def forward(self, obs):
                # Extract features using the policy's feature extractor
                features = self.policy.extract_features(obs)
                # Get action distribution (logits)
                action_logits = self.policy.action_net(features)
                return action_logits
        
        wrapped_model = ActorWrapper(policy)
        wrapped_model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset_version,
            dynamic_axes={
                input_name: {0: 'batch_size'},
                output_name: {0: 'batch_size'}
            },
            verbose=verbose,
            export_params=True,
            do_constant_folding=True
        )
        
    except Exception as e:
        if verbose:
            print(f"Standard export failed: {e}")
            print("Trying alternative export method...")
        
        # Alternative: Use the policy's _predict method
        try:
            class PolicyWrapper(torch.nn.Module):
                def __init__(self, policy):
                    super().__init__()
                    self.policy = policy
                
                def forward(self, obs):
                    # Use the policy's forward method to get action logits
                    latent_pi, _ = self.policy.mlp_extractor(obs)
                    action_logits = self.policy.action_net(latent_pi)
                    return action_logits
            
            wrapped_model = PolicyWrapper(policy)
            wrapped_model.eval()
            
            torch.onnx.export(
                wrapped_model,
                dummy_input,
                output_path,
                input_names=[input_name],
                output_names=[output_name],
                opset_version=opset_version,
                dynamic_axes={
                    input_name: {0: 'batch_size'},
                    output_name: {0: 'batch_size'}
                },
                verbose=verbose,
                export_params=True,
                do_constant_folding=True
            )
        
        except Exception as e2:
            raise RuntimeError(f"Failed to export model to ONNX: {e2}")
    
    if verbose:
        print(f"ONNX model saved to {output_path}")
    
    # Verify the ONNX model
    verify_onnx_model(output_path, input_shape, verbose=verbose)
    
    return output_path


def verify_onnx_model(
    model_path: str,
    input_shape: Tuple[int, ...] = (362,),  # Updated for candlestick sequence
    verbose: bool = True
) -> bool:
    """
    Verify that the exported ONNX model is valid and can be loaded.
    
    Args:
        model_path: Path to the ONNX model
        input_shape: Expected input shape
        verbose: Whether to print verification details
        
    Returns:
        True if model is valid, False otherwise
    """
    if not ONNX_AVAILABLE:
        if verbose:
            print("ONNX libraries not available. Cannot verify model.")
        return False
    
    try:
        # Load and check the ONNX model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        if verbose:
            print("✓ ONNX model is valid")
            
            # Print model information
            print("\nModel Information:")
            print(f"  IR Version: {onnx_model.ir_version}")
            print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
            
            # Print input/output information
            print("\nInputs:")
            for input_tensor in onnx_model.graph.input:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                        for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"  {input_tensor.name}: shape={shape}, dtype={input_tensor.type.tensor_type.elem_type}")
            
            print("\nOutputs:")
            for output_tensor in onnx_model.graph.output:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                        for dim in output_tensor.type.tensor_type.shape.dim]
                print(f"  {output_tensor.name}: shape={shape}, dtype={output_tensor.type.tensor_type.elem_type}")
        
        # Test inference with ONNX Runtime
        if verbose:
            print("\nTesting inference with ONNX Runtime...")
        
        ort_session = ort.InferenceSession(model_path)
        
        # Create test input
        test_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input})
        
        if verbose:
            print(f"✓ Inference successful")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output sample: {outputs[0][0]}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Model verification failed: {e}")
        return False


def get_model_input_output_names(model_path: str) -> Tuple[str, str]:
    """
    Get input and output names from an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        Tuple of (input_name, output_name)
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX libraries are not available.")
    
    onnx_model = onnx.load(model_path)
    
    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    
    return input_name, output_name


def create_mql5_integration_example(
    onnx_model_path: str,
    output_file: str = "mql5_integration_example.mq5"
) -> None:
    """
    Create an example MQL5 code snippet for ONNX model integration.
    
    Args:
        onnx_model_path: Path to the ONNX model
        output_file: Path to save the MQL5 example file
    """
    input_name, output_name = get_model_input_output_names(onnx_model_path)
    
    mql5_code = f"""
// MQL5 Integration Example for XAUUSD Trading System
// Generated automatically - modify as needed

#property copyright "AI Trading System"
#property version   "1.00"

#include <Trade\\Trade.mqh>

// ONNX Model Configuration
string ONNX_MODEL_PATH = "models\\\\trading_model.onnx";
string ONNX_INPUT_NAME = "{input_name}";
string ONNX_OUTPUT_NAME = "{output_name}";
long ONNX_HANDLE = INVALID_HANDLE;

// Trading Parameters
double LOT_SIZE = 0.01;
int MAX_POSITIONS = 5;

// State Space: 362 dimensions
// [Sequence of 50 candles × 7 features (350), 
//  Time features (6), Portfolio state (6)]
// Each candle: Body Size, Upper Wick, Lower Wick, Price Change, 
//              Price Change %, Candle Direction, Body to Range
int STATE_SIZE = 362;

// Action Space: 6 discrete actions
// 0: Hold, 1: Initial Buy, 2: Initial Sell, 
// 3: Recovery Buy, 4: Recovery Sell, 5: Close All
int ACTION_SIZE = 6;

// Global Variables
CTrade trade;
double state_buffer[19];
float input_buffer[19];
long output_buffer[6];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{{
    // Initialize ONNX model
    ONNX_HANDLE = OnnxCreateFromBuffer(ONNX_MODEL_PATH, ONNX_FILE);
    
    if(ONNX_HANDLE == INVALID_HANDLE)
    {{
        Print("Failed to load ONNX model: ", GetLastError());
        return(INIT_FAILED);
    }}
    
    // Set input/output shapes
    if(!OnnxSetInputShape(ONNX_HANDLE, 0, {{1, STATE_SIZE}}))
    {{
        Print("Failed to set input shape: ", GetLastError());
        OnnxRelease(ONNX_HANDLE);
        return(INIT_FAILED);
    }}
    
    if(!OnnxSetOutputShape(ONNX_HANDLE, 0, {{1, ACTION_SIZE}}))
    {{
        Print("Failed to set output shape: ", GetLastError());
        OnnxRelease(ONNX_HANDLE);
        return(INIT_FAILED);
    }}
    
    Print("ONNX model loaded successfully");
    return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    if(ONNX_HANDLE != INVALID_HANDLE)
    {{
        OnnxRelease(ONNX_HANDLE);
    }}
}}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{{
    // Prepare state observation
    PrepareState();
    
    // Run ONNX inference
    if(!RunInference())
    {{
        Print("Inference failed");
        return;
    }}
    
    // Get action from output
    int action = GetAction();
    
    // Execute trading action
    ExecuteAction(action);
}}

//+------------------------------------------------------------------+
//| Prepare state observation                                        |
//+------------------------------------------------------------------+
void PrepareState()
{{
    // Get current bar data
    double open[], high[], low[], close[];
    ArraySetAsSeries(open, true);
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);
    ArraySetAsSeries(close, true);
    
    CopyOpen(_Symbol, PERIOD_M15, 0, 1, open);
    CopyHigh(_Symbol, PERIOD_M15, 0, 1, high);
    CopyLow(_Symbol, PERIOD_M15, 0, 1, low);
    CopyClose(_Symbol, PERIOD_M15, 0, 1, close);
    
    // TODO: Build sequence of last 50 candles
    // For each candle, calculate:
    // - Body Size = abs(close - open)
    // - Upper Wick = high - max(open, close)
    // - Lower Wick = min(open, close) - low
    // - Price Change = close - previous_close
    // - Price Change % = price_change / previous_close
    // - Candle Direction = (close > open) ? 1 : ((close < open) ? -1 : 0)
    // - Body to Range = body_size / (high - low)
    // 
    // Then add time features (hour_sin, hour_cos, day_of_week, sessions)
    // And portfolio state (balance, equity, floating_pnl, positions, avg_entry, drawdown)
    
    // Example for current candle (you need to do this for last 50 candles):
    // double body_size = MathAbs(close[0] - open[0]);
    // double upper_wick = high[0] - MathMax(open[0], close[0]);
    // double lower_wick = MathMin(open[0], close[0]) - low[0];
    // ... etc
    
    // Convert to float array for ONNX
    for(int i = 0; i < STATE_SIZE; i++)
    {{
        input_buffer[i] = (float)state_buffer[i];
    }}
}}

//+------------------------------------------------------------------+
//| Run ONNX inference                                                |
//+------------------------------------------------------------------+
bool RunInference()
{{
    if(ONNX_HANDLE == INVALID_HANDLE)
        return false;
    
    // Run inference
    if(!OnnxRun(ONNX_HANDLE, ONNX_RUNMODE_NORMAL, input_buffer, output_buffer))
    {{
        Print("OnnxRun failed: ", GetLastError());
        return false;
    }}
    
    return true;
}}

//+------------------------------------------------------------------+
//| Get action from ONNX output                                      |
//+------------------------------------------------------------------+
int GetAction()
{{
    // Find action with highest probability (argmax)
    int best_action = 0;
    float best_value = output_buffer[0];
    
    for(int i = 1; i < ACTION_SIZE; i++)
    {{
        if(output_buffer[i] > best_value)
        {{
            best_value = output_buffer[i];
            best_action = i;
        }}
    }}
    
    return best_action;
}}

//+------------------------------------------------------------------+
//| Execute trading action                                           |
//+------------------------------------------------------------------+
void ExecuteAction(int action)
{{
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    switch(action)
    {{
        case 0: // Hold
            break;
            
        case 1: // Initial Buy
            if(PositionsTotal() < MAX_POSITIONS)
            {{
                trade.Buy(LOT_SIZE, _Symbol);
            }}
            break;
            
        case 2: // Initial Sell
            if(PositionsTotal() < MAX_POSITIONS)
            {{
                trade.Sell(LOT_SIZE, _Symbol);
            }}
            break;
            
        case 3: // Recovery Buy
            // TODO: Implement recovery logic
            if(PositionsTotal() < MAX_POSITIONS)
            {{
                trade.Buy(LOT_SIZE, _Symbol);
            }}
            break;
            
        case 4: // Recovery Sell
            // TODO: Implement recovery logic
            if(PositionsTotal() < MAX_POSITIONS)
            {{
                trade.Sell(LOT_SIZE, _Symbol);
            }}
            break;
            
        case 5: // Close All
            for(int i = PositionsTotal() - 1; i >= 0; i--)
            {{
                ulong ticket = PositionGetTicket(i);
                if(ticket > 0)
                {{
                    trade.PositionClose(ticket);
                }}
            }}
            break;
    }}
}}
"""
    
    with open(output_file, 'w') as f:
        f.write(mql5_code)
    
    print(f"MQL5 integration example saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export SB3 PPO model to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved SB3 model (.zip file)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/trading_model.onnx",
        help="Path to save the ONNX model"
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs='+',
        default=[362],
        help="Input shape (default: [362] for 50-candle sequence)"
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="observations",
        help="Input layer name (default: observations)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="actions",
        help="Output layer name (default: actions)"
    )
    parser.add_argument(
        "--create_mql5_example",
        action="store_true",
        help="Create MQL5 integration example"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    
    # Export model
    print("Exporting model to ONNX...")
    export_ppo_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        input_shape=tuple(args.input_shape),
        input_name=args.input_name,
        output_name=args.output_name,
        verbose=True
    )
    
    # Create MQL5 example if requested
    if args.create_mql5_example:
        create_mql5_integration_example(args.output_path)

