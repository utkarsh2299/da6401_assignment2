# import numpy as np
import sympy as sp

def calculate_complexity():
    """
    Calculate the theoretical complexity of the CNN model in terms of computations and parameters.
    
    Returns:
        tuple: (computational_complexity, parameter_count)
    """
    # Define symbolic variables
    m, k, n = sp.symbols('m k n')
    
    # Initial image size (assume 224x224 RGB images)
    height, width = 224, 224
    in_channels = 3
    num_classes = 10  # Placeholder, will be replaced with actual number
    
    total_computations = 0
    total_parameters = 0
    
    # For each of the 5 convolutional blocks
    for i in range(5):
        # Input channels for current layer
        curr_in_channels = in_channels if i == 0 else m
        
        # Computations for convolution: H*W*C_in*C_out*k*k
        conv_computations = height * width * curr_in_channels * m * k * k
        
        # Parameters for convolution: C_in*C_out*k*k + C_out (weights + bias)
        conv_parameters = curr_in_channels * m * k * k + m
        
        # After maxpooling, height and width are halved
        height //= 2
        width //= 2
        
        total_computations += conv_computations
        total_parameters += conv_parameters
    
    # Final feature map size after 5 pooling operations (224 -> 112 -> 56 -> 28 -> 14 -> 7)
    feature_map_size = height * width * m
    
    # Dense layer: input_size * output_size computations
    dense_computations = feature_map_size * n
    dense_parameters = feature_map_size * n + n  # weights + bias
    
    # Output layer: dense_neurons * num_classes computations
    output_computations = n * num_classes
    output_parameters = n * num_classes + num_classes  # weights + bias
    
    total_computations += dense_computations + output_computations
    total_parameters += dense_parameters + output_parameters
    
    # Simplify expressions
    total_computations = sp.simplify(total_computations)
    total_parameters = sp.simplify(total_parameters)
    
    return total_computations, total_parameters

def print_complexity_analysis():
    """Print the complexity analysis in a readable format."""
    computations, parameters = calculate_complexity()
    
    print("=" * 80)
    print("CNN Model Complexity Analysis")
    print("=" * 80)
    print("Variables:")
    print("  m = number of filters in each convolution layer")
    print("  k = size of convolution kernel (k×k)")
    print("  n = number of neurons in the dense layer")
    print("\nTotal Computational Complexity:")
    print(f"  {computations}")
    print("\nTotal Number of Parameters:")
    print(f"  {parameters}")
    print("=" * 80)
    
    # Example with specific values
    m_val, k_val, n_val = 32, 3, 128
    comp_val = computations.subs({'m': m_val, 'k': k_val, 'n': n_val})
    param_val = parameters.subs({'m': m_val, 'k': k_val, 'n': n_val})
    
    print(f"Example with m={m_val}, k={k_val}, n={n_val}:")
    print(f"  Computations: {int(comp_val):,}")
    print(f"  Parameters: {int(param_val):,}")
    print("=" * 80)

if __name__ == "__main__":
    print_complexity_analysis()