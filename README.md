# NOA---Neural-Orchestrator-Agent
Give any LLM or AI a brain


Using 

https://huggingface.co/Qwen/Qwen3-0.6B

https://github.com/mlech26l/ncps/blob/master/README.md

https://github.com/raminmh/CfC/blob/main/README.md

# (Full script name as per user's path: C:\Users\ali_z\ANU AI\V1\NOA\qwen3.py)

import subprocess
import sys
import os
import json 
import time
import threading
import queue
from collections import deque
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

# --- Dependency Installation ---
def ensure_package_installed(package_name_with_spec):
    """
    Checks if a package is installed according to the spec (e.g., "transformers>=4.51.0"),
    and if not, attempts to install or upgrade it.
    Returns True if the package is available and meets spec, False otherwise.
    """
    parts = package_name_with_spec.replace('>', '=').replace('<', '=').split('=')
    package_name = parts[0].strip()
    required_version_str = None
    operator = None

    if len(parts) > 1 and parts[1] == '': # Handles "package>="
        operator = ">="
        required_version_str = parts[2].strip() if len(parts) > 2 else None
    elif len(parts) > 1: # Handles "package==version" or "package>=version"
        full_spec = package_name_with_spec
        if ">=" in full_spec:
            operator = ">="
            package_name, required_version_str = map(str.strip, full_spec.split(">=", 1))
        elif "==" in full_spec:
            operator = "=="
            package_name, required_version_str = map(str.strip, full_spec.split("==", 1))
        # Add other operators if needed
    
    install_target = package_name_with_spec # Use the full spec for pip install

    try:
        module = __import__(package_name)
        print(f"Package '{package_name}' is imported.")
        if required_version_str and operator:
            # Import packaging here, only if needed and after it's potentially installed
            from packaging.version import parse as parse_version
            current_version = parse_version(module.__version__)
            required_version = parse_version(required_version_str)
            
            version_ok = False
            if operator == ">=":
                version_ok = current_version >= required_version
            elif operator == "==":
                version_ok = current_version == required_version
            # Add other operator checks if needed

            if version_ok:
                print(f"'{package_name}' version {current_version} meets requirement '{operator}{required_version_str}'.")
                return True
            else:
                print(f"'{package_name}' version {current_version} does not meet requirement '{operator}{required_version_str}'. Attempting upgrade...")
                # Fall through to install/upgrade
        else:
            # No specific version, just presence is enough
            return True

    except ImportError:
        print(f"Package '{package_name}' not found.")
    except AttributeError: # module may not have __version__ (rare for these packages)
        print(f"Could not determine version for '{package_name}'. Assuming it needs (re)installation to meet spec.")
        
    print(f"Attempting to install/upgrade '{install_target}'...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_target])
        print(f"Successfully installed/upgraded '{install_target}'.")
        if package_name == "transformers":
             print("Transformers updated. If in an interactive environment (like Jupyter), you might need to restart the kernel.")
        return True
    except Exception as e:
        print(f"Error installing/upgrading '{install_target}': {e}")
        print(f"Please install it manually: pip install \"{install_target}\"")
        return False

# Ensure required packages are installed
# packaging needs to be first if used within ensure_package_installed for others
required_packages_specs = [
    "packaging",   # For version comparison
    "torch",
    "transformers>=4.51.0",
    "accelerate", # For device_map="auto" and potentially faster loading
    "scikit-learn", # For PCA
    "matplotlib" # For plotting
]

all_packages_ready = True
for pkg_spec in required_packages_specs:
    if not ensure_package_installed(pkg_spec):
        all_packages_ready = False

if not all_packages_ready:
    print("\nOne or more required packages could not be installed or configured correctly. Please address the issues above and try again.")
    sys.exit(1)

# Now import the modules after ensuring they are installed
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import PCA 
    import matplotlib.pyplot as plt 
except ImportError as e:
    print(f"Failed to import necessary libraries even after installation attempt: {e}")
    sys.exit(1)

# --- Core Neural Network Components (NCP & CfC related) ---

class LTCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights for current input and recurrent input are combined
        self.w = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size)) # Bias
        
        # Time constant tau, Reversal potential E_rev, Leak potential v_leak
        self.tau = nn.Parameter(torch.Tensor(hidden_size)) 
        self.E_rev = nn.Parameter(torch.Tensor(hidden_size)) # In NCP paper, E is used for sensory inputs
        self.v_leak = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)
        # Initialize tau to be positive, e.g., around 1.0 to 20.0 ms (scaled)
        # NCP paper uses tau_mean=1, tau_std=0 so all tau=1 init
        nn.init.constant_(self.tau, 5.0) # A slightly larger default time constant
        nn.init.constant_(self.E_rev, 1.0) # Could be different for excitatory/inhibitory types
        nn.init.zeros_(self.v_leak)
    
    def forward(self, input_tensor: torch.Tensor, hx: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        # input_tensor: [batch, input_size_for_this_cell]
        # hx: [batch, hidden_size_of_this_cell] (previous state)
        
        # Concatenate current input and previous hidden state for matrix multiplication
        combined_input_and_state = torch.cat((input_tensor, hx), dim=-1)
        
        # Calculate total weighted input (synaptic current + bias)
        # I_syn = W_in * x + W_rec * h_prev + b
        total_weighted_input = torch.matmul(combined_input_and_state, self.w.t()) + self.b
        
        # LTC dynamics based on Kirschfeld & Pischke (1991) formulation in NCP paper:
        # dv/dt = (- (v - v_leak) + G_syn * (E_rev - v)) / tau
        # Where G_syn is related to total_weighted_input, and v is hx.
        # For simplicity, a common formulation: dv/dt = (-v_m + v_leak + I_total) / tau
        # Here, I_total can be considered total_weighted_input. E_rev is often part of I_total.
        
        # Using the NCP paper's simplified continuous-time neuron dynamics:
        # dv/dt = (-v + weighted_sensory_inputs) / tau
        # Here, total_weighted_input includes both sensory and recurrent parts.
        # The E_rev and v_leak make it more like a Leaky Integrate-and-Fire neuron's subthreshold dynamics.
        
        # Let's use a formulation closer to the standard leaky integrator driven by total_weighted_input:
        # tau * dv/dt = -(v - v_leak) + I_eff
        # where I_eff could be total_weighted_input. The E_rev term implies conductance changes.
        # A common simplification: tau * dv/dt = -v + I_total_effective
        
        # Clamping tau to be positive
        positive_tau = torch.relu(self.tau) + 1e-3 # Add epsilon for stability

        # dv_dt = (self.v_leak - hx + total_weighted_input) / positive_tau # Simple leaky integrator
        # Let's make it more like: tau dv/dt = -(v - v_leak) + R*I where I is total_weighted_input.
        # Or dv/dt = (v_leak - hx)/tau + total_weighted_input / C (where tau = R*C) -- need to be careful with units/scaling
        
        # Simplest robust form often used:
        dv_dt = (-hx + total_weighted_input) / positive_tau # total_weighted_input acts as the target potential scaled by R
        
        new_hx = hx + dv_dt * dt
        return torch.tanh(new_hx) # Apply activation function to the state for output

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size: int):
        super(AttentionMechanism, self).__init__()
        # This version scales each hidden unit based on its own value
        self.attention_net = nn.Linear(hidden_size, hidden_size) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # x is [batch, features] or [batch, seq, features]
        if x.dim() not in [2, 3]:
            raise ValueError("Input tensor must be 2D [batch, features] or 3D [batch, seq, features]")

        original_shape = x.shape
        if x.dim() == 3: # [batch, seq_len, features]
            x_reshaped = x.reshape(-1, original_shape[-1]) # [batch*seq_len, features]
        else: # [batch, features]
            x_reshaped = x
            
        attention_signal = self.attention_net(x_reshaped) # [batch*seq_len or batch, features]
        attention_weights = torch.softmax(attention_signal, dim=-1) # Softmax over feature dimension
        
        attended_x_reshaped = x_reshaped * attention_weights # Element-wise scaling
        
        if x.dim() == 3:
            return attended_x_reshaped.reshape(original_shape)
        else:
            return attended_x_reshaped

class NCP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(NCP, self).__init__()
        if len(hidden_sizes) != 3:
            raise ValueError("hidden_sizes must be a list of 3 integers for sensory, inter, command layers.")
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes # [sensory_h, inter_h, command_h]
        self.output_size = output_size   # motor_h (output_size)
        
        # Added: Initialize activation history BEFORE reset_parameters is called
        self.activation_history = {
            'sensory': [],
            'inter': [],
            'command': [],
            'motor': []
        }
        
        # Layer definitions using LTCCell
        self.sensory_neurons = LTCCell(input_size, hidden_sizes[0])
        self.inter_neurons = LTCCell(hidden_sizes[0], hidden_sizes[1]) 
        # Command neurons receive input from inter-neurons and recurrently from themselves
        self.command_neurons = LTCCell(hidden_sizes[1] + hidden_sizes[2], hidden_sizes[2]) 
        
        # FIXED LINE: Changed input_size from hidden_sizes[2] to output_size
        # This ensures the motor neurons expect input of size output_size (after transformation)
        self.motor_neurons = LTCCell(output_size, output_size)

        self.use_attention = True # Configurable
        if self.use_attention:
            self.sensory_attention = AttentionMechanism(hidden_sizes[0])
            self.inter_attention = AttentionMechanism(hidden_sizes[1])
            self.command_attention = AttentionMechanism(hidden_sizes[2])
        
        # Explicit inter-layer weight matrices (as per user's ncp.py implies)
        # These are distinct from the internal weights of LTCCells.
        self.sensory_to_inter_w = nn.Parameter(torch.Tensor(hidden_sizes[0], hidden_sizes[1]))
        self.inter_to_command_w = nn.Parameter(torch.Tensor(hidden_sizes[1], hidden_sizes[1])) # Inter output to inter_input part of command
        self.command_recurrent_w = nn.Parameter(torch.Tensor(hidden_sizes[2], hidden_sizes[2])) # Command output to recurrent_input part of command
        self.command_to_motor_w = nn.Parameter(torch.Tensor(hidden_sizes[2], output_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize LTCCells (they have their own reset_parameters)
        for module in self.modules():
            if isinstance(module, LTCCell):
                module.reset_parameters()
        
        # Initialize explicit inter-layer weights
        nn.init.xavier_uniform_(self.sensory_to_inter_w)
        nn.init.xavier_uniform_(self.inter_to_command_w)
        nn.init.xavier_uniform_(self.command_recurrent_w)
        nn.init.xavier_uniform_(self.command_to_motor_w)
        
        # Optional: Apply sparsity as in user's code, though xavier_uniform_ might be better for dense init
        # nn.init.sparse_(self.sensory_to_inter_w, sparsity=0.5) 
        # ... and for others
        
        # Clear activation history
        for key in self.activation_history:
            self.activation_history[key] = []

    def forward(self, input_tensor: torch.Tensor, 
                hidden_states_list: List[torch.Tensor], dt: float = 0.1,
                store_activations: bool = False
               ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # hidden_states_list: [sensory_hx, inter_hx, command_hx, motor_hx]
        if len(hidden_states_list) != 4:
            raise ValueError("NCP expects 4 hidden states (sensory, inter, command, motor).")

        sensory_hx, inter_hx, command_hx, motor_hx = hidden_states_list

        # Sensory Layer
        new_sensory_hx = self.sensory_neurons(input_tensor, sensory_hx, dt)
        sensory_output_signal = new_sensory_hx
        if self.use_attention: sensory_output_signal = self.sensory_attention(sensory_output_signal)
        
        # Inter Layer
        inter_input_signal = torch.matmul(sensory_output_signal, self.sensory_to_inter_w)
        new_inter_hx = self.inter_neurons(inter_input_signal, inter_hx, dt)
        inter_output_signal = new_inter_hx
        if self.use_attention: inter_output_signal = self.inter_attention(inter_output_signal)

        # Command Layer
        command_feedforward_input = torch.matmul(inter_output_signal, self.inter_to_command_w)
        command_recurrent_feedback = torch.matmul(command_hx, self.command_recurrent_w) # Use previous command state for recurrence
        command_total_input_signal = torch.cat((command_feedforward_input, command_recurrent_feedback), dim=-1)
        new_command_hx = self.command_neurons(command_total_input_signal, command_hx, dt)
        command_output_signal = new_command_hx
        if self.use_attention: command_output_signal = self.command_attention(command_output_signal)

        # Motor Layer
        motor_input_signal = torch.matmul(command_output_signal, self.command_to_motor_w)
        new_motor_hx = self.motor_neurons(motor_input_signal, motor_hx, dt)
        motor_output_signal = new_motor_hx # This is the final output of NCP
        
        updated_hidden_states = [
            new_sensory_hx.detach(), 
            new_inter_hx.detach(), 
            new_command_hx.detach(), 
            new_motor_hx.detach()
        ]
        
        # Added: Store activations if requested
        if store_activations:
            self.activation_history['sensory'].append(new_sensory_hx.detach().cpu())
            self.activation_history['inter'].append(new_inter_hx.detach().cpu())
            self.activation_history['command'].append(new_command_hx.detach().cpu())
            self.activation_history['motor'].append(new_motor_hx.detach().cpu())
            
        return motor_output_signal, updated_hidden_states
    
    # Added: Method to visualize activation patterns
    def visualize_activations(self, layer_name=None, time_steps=None):
        """Visualize activation patterns of specified layer(s) over time using numpy and matplotlib."""
        if not any(self.activation_history.values()):
            print("No activation history available. Run forward pass with store_activations=True first.")
            return None
            
        if layer_name and layer_name not in self.activation_history:
            raise ValueError(f"Invalid layer name. Choose from: {list(self.activation_history.keys())}")
            
        layers_to_plot = [layer_name] if layer_name else self.activation_history.keys()
        
        # Determine time steps to plot
        if time_steps is None:
            # Get max length of any activation history
            max_steps = max([len(self.activation_history[l]) for l in layers_to_plot if self.activation_history[l]])
            time_steps = range(max_steps)
        
        fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(10, 3*len(layers_to_plot)))
        if len(layers_to_plot) == 1:
            axes = [axes]  # Make it iterable for single layer case
            
        for i, layer in enumerate(layers_to_plot):
            if not self.activation_history[layer]:
                axes[i].text(0.5, 0.5, f"No activation data for {layer}", 
                             horizontalalignment='center', verticalalignment='center')
                continue
                
            # Convert activations to numpy for easier manipulation
            activations = [a.numpy().squeeze() for a in self.activation_history[layer]]
            
            # Create a 2D array where rows are time steps and columns are neurons
            act_array = np.array(activations)
            if len(act_array.shape) > 2:  # Handle case where squeeze didn't remove all extra dimensions
                act_array = act_array.reshape(act_array.shape[0], -1)
                
            # Plot heatmap of activations over time
            im = axes[i].imshow(act_array, aspect='auto', cmap='viridis')
            axes[i].set_title(f"{layer.capitalize()} Layer Activations Over Time")
            axes[i].set_xlabel("Neuron Index")
            axes[i].set_ylabel("Time Step")
            fig.colorbar(im, ax=axes[i], label="Activation Value")
            
        plt.tight_layout()
        return fig

class CfCModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CfCModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize state history before other components
        self.state_history = []
        
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.f_net = nn.Linear(hidden_size, hidden_size) # Gate f
        self.g_net = nn.Linear(hidden_size, hidden_size) # Candidate state g
        self.h_net = nn.Linear(hidden_size, hidden_size) # Candidate state h (or previous state influence)
        
    def forward(self, x_input: torch.Tensor, t_time_delta: torch.Tensor, prev_hidden_state: torch.Tensor, store_history: bool = False):
        # x_input: [batch, input_size]
        # t_time_delta: [batch, 1] or scalar, represents time delta
        # prev_hidden_state: [batch, hidden_size]
        
        combined = torch.cat([x_input, prev_hidden_state], dim=-1)
        backbone_out = self.backbone(combined)
        
        f = self.f_net(backbone_out)  # Gate parameter
        g = self.g_net(backbone_out)  # Transform for new input
        h = self.h_net(backbone_out)  # Transform for decay/previous state influence
        
        # CfC update rule: new_hidden = sigmoid(-f*t) * g + (1 - sigmoid(-f*t)) * h
        # This looks like a form of gated update.
        # If h is prev_hidden_state, it's like: new_h = gate * new_info_g + (1-gate) * old_info_h
        sigma_ft = torch.sigmoid(-f * t_time_delta) 
        
        # The original formulation: new_hidden = σ(-f*Δt) * g(x,h_prev) + (1 - σ(-f*Δt)) * h(x,h_prev)
        # This means g and h are candidates derived from current input and prev_hidden.
        # The formulation in user's cfc.py seems to be this.
        new_hidden_state = sigma_ft * g + (1 - sigma_ft) * h
        
        # Added: Store state in history if requested
        if store_history:
            self.state_history.append(new_hidden_state.detach().cpu())
            
        return new_hidden_state
    
    # Added: Reset state history
    def reset_state_history(self):
        self.state_history = []
        
    # Added: Visualize state evolution
    def visualize_state_evolution(self, time_window=None, n_neurons=None):
        """Visualize the evolution of hidden state over time using numpy and matplotlib."""
        if not self.state_history:
            print("No state history available. Run forward pass with store_history=True first.")
            return None
            
        # Convert to numpy for easier manipulation
        states = [s.numpy().squeeze() for s in self.state_history]
        state_array = np.array(states)
        
        # Determine time window to plot
        if time_window is None:
            time_window = range(len(state_array))
        else:
            time_window = range(min(time_window, len(state_array)))
            
        # Determine which neurons to plot
        if n_neurons is None or n_neurons >= state_array.shape[1]:
            neuron_indices = range(state_array.shape[1])
        else:
            # Select neurons with highest variance (most informative)
            var_per_neuron = np.var(state_array, axis=0)
            neuron_indices = np.argsort(-var_per_neuron)[:n_neurons]  # Negative to sort descending
            
        plt.figure(figsize=(12, 6))
        for idx in neuron_indices:
            plt.plot(time_window, state_array[time_window, idx], label=f'Neuron {idx}')
            
        plt.title('CfC Hidden State Evolution Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Activation Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()

# --- Controller and Learning Modules ---

class NCPController(nn.Module):
    def __init__(self, controller_input_size, ncp_hidden_sizes: List[int], ncp_output_size: int, l1_lambda=0.001):
        super(NCPController, self).__init__()
        self.controller_input_size = controller_input_size
        self.ncp_hidden_sizes = ncp_hidden_sizes 
        self.ncp_output_size = ncp_output_size 
        self.l1_lambda = l1_lambda
        
        # Added: Initialize control signal history before other components
        self.control_signal_history = []

        self.ncp = NCP(controller_input_size, ncp_hidden_sizes, ncp_output_size)
        self.ncp_internal_hidden_states = None # Initialized in reset_hidden_states
        self.reset_hidden_states()

    def forward(self, combined_input: torch.Tensor, dt: float = 0.1, store_activations: bool = False, store_control_signals: bool = False):
        # combined_input: [batch, controller_input_size]
        if self.ncp_internal_hidden_states is None or combined_input.shape[0] != self.ncp_internal_hidden_states[0].shape[0]:
             self.reset_hidden_states(batch_size=combined_input.shape[0], device=combined_input.device)

        # Pad or truncate combined_input if its feature dimension doesn't match ncp.input_size
        if combined_input.size(1) != self.ncp.input_size:
            if combined_input.size(1) < self.ncp.input_size:
                padding = torch.zeros(combined_input.size(0), self.ncp.input_size - combined_input.size(1), device=combined_input.device)
                combined_input = torch.cat([combined_input, padding], dim=1)
            else:
                combined_input = combined_input[:, :self.ncp.input_size]
        
        control_signals, new_ncp_hidden_states = self.ncp(
            combined_input, 
            self.ncp_internal_hidden_states, 
            dt,
            store_activations=store_activations
        )
        
        self.ncp_internal_hidden_states = new_ncp_hidden_states
        
        # Added: Store control signals if requested
        if store_control_signals:
            self.control_signal_history.append(control_signals.detach().cpu())
            
        return control_signals

    def reset_hidden_states(self, batch_size=1, device='cpu'):
        # [sensory_hx, inter_hx, command_hx, motor_hx]
        self.ncp_internal_hidden_states = [
            torch.zeros(batch_size, self.ncp_hidden_sizes[0], device=device),
            torch.zeros(batch_size, self.ncp_hidden_sizes[1], device=device),
            torch.zeros(batch_size, self.ncp_hidden_sizes[2], device=device),
            torch.zeros(batch_size, self.ncp_output_size, device=device) # Motor layer output size
        ]
        
        # Reset control signal history
        self.control_signal_history = []

    def get_l1_loss(self):
        l1_loss = torch.tensor(0.0, device=next(self.ncp.parameters()).device) # Ensure loss is on correct device
        for param in self.ncp.parameters(): 
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss
    
    # Added: Visualize control signals over time
    def visualize_control_signals(self, time_window=None, save_path=None):
        """Visualize the evolution of control signals over time using numpy and matplotlib."""
        if not self.control_signal_history:
            print("No control signal history available. Run forward pass with store_control_signals=True first.")
            return None
            
        # Convert to numpy for easier visualization
        signals = [s.numpy().squeeze() for s in self.control_signal_history]
        signal_array = np.array(signals)
        
        # Determine time window to plot
        if time_window is None:
            time_window = range(len(signal_array))
        else:
            time_window = range(min(time_window, len(signal_array)))
            
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot control signal values over time
        for i in range(signal_array.shape[1]):
            ax1.plot(time_window, signal_array[time_window, i], label=f'Signal {i}')
            
        ax1.set_title('NCP Control Signals Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Signal Value')
        ax1.grid(True, alpha=0.3)
        
        # If we have enough time steps, show frequency analysis of the most active signals
        if len(time_window) > 10:
            # Find most active signals (highest variance)
            var_per_signal = np.var(signal_array[time_window], axis=0)
            top_signals = np.argsort(-var_per_signal)[:5]  # Top 5 most variable signals
            
            # Plot frequency domain for top signals
            for idx in top_signals:
                signal = signal_array[time_window, idx]
                # Zero-mean the signal for better FFT
                signal = signal - np.mean(signal)
                signal_fft = np.abs(np.fft.fft(signal))
                freqs = np.fft.fftfreq(len(signal))
                
                # Plot positive frequencies only (up to Nyquist frequency)
                pos_freq_idx = np.where(freqs > 0)[0]
                ax2.plot(freqs[pos_freq_idx], signal_fft[pos_freq_idx], label=f'Signal {idx}')
                
            ax2.set_title('Frequency Domain Analysis of Top Control Signals')
            ax2.set_xlabel('Frequency')
            ax2.set_ylabel('Magnitude')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Not enough time steps for frequency analysis", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Add a colorbar-style legend for the first plot with many lines
        if signal_array.shape[1] > 10:
            # Replace individual line legend with a colormap legend
            handles, labels = ax1.get_legend_handles_labels()
            ax1.get_legend().remove() if ax1.get_legend() else None
            
            signal_indices = np.arange(signal_array.shape[1])
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=signal_array.shape[1]-1))
            cbar = fig.colorbar(sm, ax=ax1)
            cbar.set_label('Signal Index')
            
        else:
            ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        
        return fig

class ContinuousLearningModule(nn.Module):
    def __init__(self, clm_input_size, cfc_hidden_size):
        super(ContinuousLearningModule, self).__init__()
        self.clm_input_size = clm_input_size
        self.cfc_hidden_size = cfc_hidden_size
        self.cfc = CfCModule(clm_input_size, cfc_hidden_size)
        self.cfc_hidden_state = None # Initialized in reset_hidden_state
        
        # Initialize processing times before reset_hidden_state
        self.processing_times = []
        self.reset_hidden_state()

    def forward(self, interaction_history_features: torch.Tensor, time_step_tensor: torch.Tensor, store_history: bool = False):
        # Added: Measure processing time
        start_time = time.time()
        
        # interaction_history_features: [batch, clm_input_size]
        # time_step_tensor: [batch, 1] (delta_t or current time)
        if self.cfc_hidden_state is None or interaction_history_features.shape[0] != self.cfc_hidden_state.shape[0]:
            self.reset_hidden_state(batch_size=interaction_history_features.shape[0], device=interaction_history_features.device)

        if interaction_history_features.size(1) != self.cfc.input_size:
            if interaction_history_features.size(1) < self.cfc.input_size:
                padding = torch.zeros(interaction_history_features.size(0), self.cfc.input_size - interaction_history_features.size(1), device=interaction_history_features.device)
                interaction_history_features = torch.cat([interaction_history_features, padding], dim=1)
            else:
                interaction_history_features = interaction_history_features[:, :self.cfc.input_size]

        cfc_output_hidden = self.cfc(
            interaction_history_features, 
            time_step_tensor, 
            self.cfc_hidden_state,
            store_history=store_history
        )
        
        self.cfc_hidden_state = cfc_output_hidden.detach()
        
        # Added: Record processing time
        end_time = time.time()
        self.processing_times.append(end_time - start_time)
        
        return cfc_output_hidden

    def reset_hidden_state(self, batch_size=1, device='cpu'):
        self.cfc_hidden_state = torch.zeros(batch_size, self.cfc_hidden_size, device=device)
        self.cfc.reset_state_history()
        self.processing_times = []
        
    # Added: Get average processing time
    def get_avg_processing_time(self):
        """Return the average processing time per forward pass."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    # Added: Analyze CLM performance
    def analyze_performance(self):
        """Analyze CLM processing performance."""
        if not self.processing_times:
            return "No performance data available yet."
            
        avg_time = self.get_avg_processing_time()
        max_time = max(self.processing_times)
        min_time = min(self.processing_times)
        std_dev = np.std(self.processing_times)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(self.processing_times)
        plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Avg: {avg_time:.6f}s')
        plt.fill_between(range(len(self.processing_times)), 
                         np.array(self.processing_times) - std_dev,
                         np.array(self.processing_times) + std_dev,
                         alpha=0.2, color='red')
        plt.title('CLM Processing Time per Forward Pass')
        plt.xlabel('Forward Pass Index')
        plt.ylabel('Processing Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        performance_stats = {
            "avg_time": avg_time,
            "max_time": max_time,
            "min_time": min_time,
            "std_dev": std_dev,
            "total_forwards": len(self.processing_times)
        }
        
        return performance_stats, plt.gcf()

# --- LLM Interface (Qwen3) ---
class Qwen3Interface:
    def __init__(self, model_path_or_name="Qwen/Qwen3-0.6B", local_model_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # Added: Timing metrics
        self.loading_time = 0
        self.generation_times = []
        
        identifier_to_load = model_path_or_name
        if local_model_dir and os.path.isdir(local_model_dir):
            if os.path.exists(os.path.join(local_model_dir, "config.json")):
                print(f"Attempting to use local Qwen3 model from: {local_model_dir}")
                identifier_to_load = local_model_dir
            else:
                print(f"Warning: Local path {local_model_dir} provided but seems invalid (missing config.json).")
                print(f"Falling back to Hugging Face Hub model: {model_path_or_name}")
        else:
             print(f"Local model directory not specified or not found. Using Hub model: {model_path_or_name}")

        print(f"Loading Qwen3 model and tokenizer ('{identifier_to_load}')...")
        
        # Added: Measure loading time
        load_start_time = time.time()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(identifier_to_load)
            self.model = AutoModelForCausalLM.from_pretrained(
                identifier_to_load,
                torch_dtype="auto", 
                device_map="auto" 
            )
            
            # Added: Record loading time
            self.loading_time = time.time() - load_start_time
            
            print(f"Qwen3 model loaded successfully on device: {self.model.device} in {self.loading_time:.2f} seconds")
            # Update self.device to where the model actually loaded, if different
            if str(self.model.device) != self.device:
                print(f"Model mapped to {self.model.device}. Interface device updated accordingly.")
                self.device = str(self.model.device)

        except Exception as e:
            print(f"Error loading Qwen3 model: {e}")
            raise

        self.temperature = 0.6
        self.num_predict = 512 
        self.top_p = 0.95
        self.top_k = 20
        self.presence_penalty = 0.0 

    def adjust_parameters(self, temperature=None, num_predict=None, top_p=None, top_k=None, presence_penalty=None):
        if temperature is not None: self.temperature = temperature
        if num_predict is not None: self.num_predict = num_predict
        if top_p is not None: self.top_p = top_p
        if top_k is not None: self.top_k = top_k
        if presence_penalty is not None: self.presence_penalty = presence_penalty
            
    def generate_qwen_prompt(self, task_description: str, control_signals_text: str = None, conversation_history_list: list = None):
        messages = []
        if conversation_history_list: 
            messages.extend(conversation_history_list)
        
        current_user_content = task_description
        if control_signals_text:
            current_user_content = f"Task: {task_description}\nControl Signals Guide: {control_signals_text}\n\nBased on these, provide your response:"
        
        messages.append({"role": "user", "content": current_user_content})
        
        templated_prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=True 
        )
        return templated_prompt_text

    def generate_response(self, templated_prompt_text: str):
        # Added: Measure generation time
        gen_start_time = time.time()
        
        # Ensure model inputs are on the same device as the model
        model_inputs = self.tokenizer([templated_prompt_text], return_tensors="pt").to(self.model.device)
        
        generation_params = {
            "max_new_tokens": int(self.num_predict),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": int(self.top_k),
            "do_sample": True if self.temperature > 0 else False, # Greedy if temp is 0
            "pad_token_id": self.tokenizer.eos_token_id 
        }
        if self.presence_penalty != 0.0:
             generation_params["presence_penalty"] = self.presence_penalty

        generated_ids_tensor = self.model.generate(**model_inputs, **generation_params)
        
        input_ids_length = model_inputs.input_ids.shape[1]
        output_token_ids = generated_ids_tensor[0][input_ids_length:].tolist()

        think_end_token_id = 151668 
        thinking_content_str = ""
        final_content_str = ""

        try:
            reversed_output_ids = output_token_ids[::-1]
            idx_of_tag_in_reversed = reversed_output_ids.index(think_end_token_id)
            split_index_after_think_tag = len(output_token_ids) - idx_of_tag_in_reversed
            
            thinking_tokens_segment = output_token_ids[:split_index_after_think_tag]
            thinking_content_str = self.tokenizer.decode(thinking_tokens_segment, skip_special_tokens=True).strip()
            if thinking_content_str.startswith("<think>"): thinking_content_str = thinking_content_str[len("<think>"):]
            if thinking_content_str.endswith("</think>"): thinking_content_str = thinking_content_str[:-len("</think>")].strip()


            content_tokens_segment = output_token_ids[split_index_after_think_tag:]
            final_content_str = self.tokenizer.decode(content_tokens_segment, skip_special_tokens=True).strip()

        except ValueError: 
            # print("Warning: '</think>' token not found. Treating all output as final content.")
            final_content_str = self.tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()
        
        # Added: Record generation time
        gen_time = time.time() - gen_start_time
        self.generation_times.append(gen_time)
        
        return final_content_str, thinking_content_str
    
    def reset_context(self): # Not directly used if history is managed externally
        pass
    
    # Added: Get LLM performance metrics
    def get_performance_metrics(self):
        """Return performance metrics for the LLM."""
        if not self.generation_times:
            return {
                "loading_time": self.loading_time,
                "avg_generation_time": 0,
                "total_generations": 0
            }
            
        return {
            "loading_time": self.loading_time,
            "avg_generation_time": sum(self.generation_times) / len(self.generation_times),
            "max_generation_time": max(self.generation_times),
            "min_generation_time": min(self.generation_times),
            "total_generations": len(self.generation_times)
        }
    
    # Added: Visualize LLM generation performance
    def visualize_performance(self):
        """Generate performance visualization for LLM generations."""
        if not self.generation_times:
            return "No generation data available yet."
            
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(self.generation_times)
        avg_time = sum(self.generation_times) / len(self.generation_times)
        plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Avg: {avg_time:.2f}s')
        plt.title('LLM Generation Time per Request')
        plt.xlabel('Request Index')
        plt.ylabel('Generation Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()

# --- Performance Monitor and Fine-tuning Data Collector ---
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        # Added: Track metrics over time
        self.metrics_history = []
        # Added: Track processing times
        self.evaluation_times = []
    
    def evaluate(self, response: str, task: str):
        # Added: Measure evaluation time
        start_time = time.time()
        
        self.metrics = {}
        self.metrics['response_length'] = len(response)
        self.metrics['relevance'] = self._compute_relevance(response, task)
        
        # Added: Compute additional metrics
        self.metrics['token_count'] = len(response.split())
        self.metrics['response_complexity'] = self._compute_complexity(response)
        
        # Added: Track metrics over time
        self.metrics_history.append(self.metrics.copy())
        
        # Added: Record evaluation time
        eval_time = time.time() - start_time
        self.evaluation_times.append(eval_time)
        
        return self.metrics.copy() # Return a copy

    def _compute_relevance(self, response, task):
        task_words = set(task.lower().split())
        response_words = set(response.lower().split())
        if not task_words: return 0.0
        return len(task_words.intersection(response_words)) / len(task_words)
    
    # Added: Compute response complexity
    def _compute_complexity(self, text):
        # Simple metric: average word length
        words = text.split()
        if not words: return 0
        return sum(len(word) for word in words) / len(words)
    
    # Added: Get metrics trend
    def get_metrics_trend(self, metric_name=None):
        """Get trends of metrics over time."""
        if not self.metrics_history:
            return "No metrics history available."
            
        if metric_name and metric_name not in self.metrics_history[0]:
            valid_metrics = list(self.metrics_history[0].keys())
            return f"Invalid metric name. Choose from: {valid_metrics}"
            
        metrics_to_plot = [metric_name] if metric_name else list(self.metrics_history[0].keys())
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4*len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]  # Make it iterable for single metric case
            
        for i, metric in enumerate(metrics_to_plot):
            values = [m.get(metric, 0) for m in self.metrics_history]
            axes[i].plot(values, marker='o')
            axes[i].set_title(f'{metric} Over Time')
            axes[i].set_xlabel('Interaction Index')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            
            # Add moving average if we have enough points
            if len(values) > 5:
                window_size = min(5, len(values)//2)
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                shift = window_size // 2  # To align moving average
                axes[i].plot(range(shift, shift + len(moving_avg)), moving_avg, 'r--', label=f'{window_size}-point MA')
                axes[i].legend()
                
        plt.tight_layout()
        return fig
    
    # Added: Reset metrics history
    def reset_metrics_history(self):
        self.metrics_history = []
        self.evaluation_times = []

class FineTuningDataCollector:
    def __init__(self, llm_interface_ref=None): 
        self.llm_interface = llm_interface_ref 
        self.fine_tuning_data = []
        self.performance_threshold = 0.6 # Lowered for more data collection initially
        self.data_file_path = "qwen3_finetuning_data.jsonl"
        
        # Added: Track data collection metrics
        self.collection_times = []
        self.rejected_points = 0
        self.accepted_points = 0

    def add_data_point(self, interaction_tuple: Tuple[str,str], performance_metrics: dict):
        # Added: Measure data collection processing time
        start_time = time.time()
        
        avg_performance = sum(performance_metrics.values()) / len(performance_metrics) if performance_metrics else 0
        
        if performance_metrics.get('relevance', 0) > self.performance_threshold or avg_performance > self.performance_threshold:
            self.fine_tuning_data.append({
                "prompt": interaction_tuple[0], 
                "completion": interaction_tuple[1],
                # Added: Include timestamp and metrics for analysis
                "timestamp": time.time(),
                "metrics": performance_metrics
            })
            self.accepted_points += 1
            print(f"Data point added for FT. Total: {len(self.fine_tuning_data)}. Relevance: {performance_metrics.get('relevance',0):.2f}")
        else:
            self.rejected_points += 1
        
        # Added: Record processing time
        self.collection_times.append(time.time() - start_time)

    def save_data_for_fine_tuning(self): 
        if not self.fine_tuning_data:
            print("No new data collected for fine-tuning.")
            return

        print(f"Saving {len(self.fine_tuning_data)} data points to {self.data_file_path}...")
        # Append to file instead of overwriting, or manage versions
        mode = 'a' if os.path.exists(self.data_file_path) else 'w'
        with open(self.data_file_path, mode) as f:
            for item in self.fine_tuning_data:
                # Remove metrics from the saved data (only used for internal analysis)
                save_item = item.copy()
                if 'metrics' in save_item:
                    del save_item['metrics']
                if 'timestamp' in save_item:
                    del save_item['timestamp']
                
                json.dump(save_item, f)
                f.write('\n')
        print(f"Fine-tuning data saved/appended. Current data buffer cleared.")
        self.fine_tuning_data = []
    
    # Added: Generate collection statistics
    def get_collection_stats(self):
        """Return statistics about the data collection process."""
        total_points = self.accepted_points + self.rejected_points
        acceptance_rate = self.accepted_points / total_points if total_points > 0 else 0
        
        return {
            "accepted_points": self.accepted_points,
            "rejected_points": self.rejected_points,
            "total_points": total_points,
            "acceptance_rate": acceptance_rate,
            "avg_processing_time": sum(self.collection_times) / len(self.collection_times) if self.collection_times else 0
        }
    
    # Added: Reset collection statistics
    def reset_collection_stats(self):
        self.collection_times = []
        self.rejected_points = 0
        self.accepted_points = 0

# --- Added: AsyncResponseGenerator for parallel processing ---
class AsyncResponseGenerator:
    def __init__(self, hybrid_brain):
        self.brain = hybrid_brain
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()  # Thread safety for shared resources
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        print("Async response generator started.")
    
    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            print("Async response generator stopped.")
    
    def _process_queue(self):
        while self.running:
            try:
                user_input, request_id = self.request_queue.get(timeout=1)
                try:
                    with self.lock:  # Ensure thread-safe access to shared brain
                        response = self.brain.process_and_respond(user_input)
                    self.response_queue.put((request_id, user_input, response, None))
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    self.response_queue.put((request_id, user_input, None, str(e) + "\n" + error_details))
                self.request_queue.task_done()
            except queue.Empty:
                continue
    
    def queue_request(self, user_input, request_id=None):
        """Queue a request for processing with an optional ID."""
        if request_id is None:
            request_id = str(time.time())  # Use timestamp as default ID
        self.request_queue.put((user_input, request_id))
        return request_id
    
    def get_response(self, block=True, timeout=None):
        """Get a processed response from the queue.
        
        Returns:
            Tuple of (request_id, user_input, response, error)
            If no response is available, returns (None, None, None, "Response not ready yet")
        """
        try:
            return self.response_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return (None, None, None, "Response not ready yet")
    
    def get_queue_status(self):
        """Return the current queue status."""
        return {
            "requests_queued": self.request_queue.qsize(),
            "responses_ready": self.response_queue.qsize()
        }

# --- Hybrid Brain Orchestrator ---
class HybridBrainQwen:
    def __init__(self, 
                 controller_input_size=192, 
                 ncp_hidden_sizes=[64, 64, 64], 
                 ncp_output_size=32, 
                 clm_input_size=128, 
                 cfc_hidden_size=32,
                 qwen_model_path_or_name="Qwen/Qwen3-0.6B",
                 qwen_local_model_dir=None,
                 lr=0.0005): # Adjusted learning rate
        
        # Determine device for NCP/CfC based on PyTorch availability
        self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"HybridBrainQwen (NCP/CfC components) operating on device: {self.compute_device}")

        self.controller_input_size = controller_input_size
        self.ncp_output_size = ncp_output_size 

        # Initialize member variables at the top for clear visibility
        self.processing_times = []
        self.learning_losses = []
        self.async_generator = None
        self.conversation_history = deque(maxlen=10) # Shorter history for faster processing

        self.ncp_controller = NCPController(controller_input_size, ncp_hidden_sizes, ncp_output_size).to(self.compute_device)
        self.llm_interface = Qwen3Interface(model_path_or_name=qwen_model_path_or_name, local_model_dir=qwen_local_model_dir)
        # LLM will be on its own device as determined by device_map="auto" in Qwen3Interface

        self.continuous_learning = ContinuousLearningModule(clm_input_size, cfc_hidden_size).to(self.compute_device)
        self.performance_monitor = PerformanceMonitor()
        self.ft_data_collector = FineTuningDataCollector()
        
        self.optimizer = torch.optim.AdamW( # Using AdamW
            list(self.ncp_controller.parameters()) + list(self.continuous_learning.parameters()), 
            lr=lr, weight_decay=0.01 # Added weight decay
        )
        self.mse_loss = nn.MSELoss()

    def _text_to_features(self, text: Union[str, List[str]], max_len: int) -> torch.Tensor:
        """Convert text to feature tensor for neural processing.
        
        Args:
            text: Either a single string or a list of strings
            max_len: Maximum length of the resulting feature tensor
            
        Returns:
            torch.Tensor: Feature representation of the text
        """
        # Handle list of strings
        if isinstance(text, list):
            text = " ".join(text)
            
        encoded = [ord(c) % 256 for c in text] # Ensure values are byte-range for embedding or direct use
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        elif len(encoded) < max_len:
            encoded.extend([0] * (max_len - len(encoded)))
        # Normalize to [0,1] or [-1,1] might be beneficial if these are direct inputs
        features_tensor = torch.tensor(encoded, dtype=torch.float32, device=self.compute_device) / 255.0 
        return features_tensor.unsqueeze(0) # [1, max_len]

    def get_combined_input_for_ncp(self, user_input_text: str) -> torch.Tensor:
        # This function prepares the single input vector for the NCP controller.
        # Example allocation: 1/3 history, 1/3 task, 1/3 performance
        part_len = self.controller_input_size // 3
        remainder = self.controller_input_size % 3

        history_text = " ".join([msg["content"] for msg in self.conversation_history])
        history_feat = self._text_to_features(history_text, part_len + (1 if remainder > 0 else 0) )

        task_feat = self._text_to_features(user_input_text, part_len + (1 if remainder > 1 else 0) )
        
        perf_values = list(self.performance_monitor.metrics.values()) if self.performance_monitor.metrics else [0.0] * 2 # Default 2 metrics
        # Ensure perf_values is a flat list of numbers
        flat_perf_values = []
        for item in perf_values:
            if isinstance(item, (list, tuple)): flat_perf_values.extend(item)
            else: flat_perf_values.append(item)
        
        perf_tensor = torch.tensor(flat_perf_values, dtype=torch.float32, device=self.compute_device)
        
        # Pad or truncate performance features
        perf_feat_target_len = part_len
        if perf_tensor.numel() < perf_feat_target_len:
            padding = torch.zeros(perf_feat_target_len - perf_tensor.numel(), device=self.compute_device)
            perf_feat = torch.cat([perf_tensor.flatten(), padding]).unsqueeze(0)
        else:
            perf_feat = perf_tensor.flatten()[:perf_feat_target_len].unsqueeze(0)
            
        ncp_input = torch.cat([history_feat, task_feat, perf_feat], dim=1)
        # Final check for total length
        if ncp_input.size(1) != self.controller_input_size:
             ncp_input = torch.nn.functional.pad(ncp_input, (0, self.controller_input_size - ncp_input.size(1))) if ncp_input.size(1) < self.controller_input_size else ncp_input[:, :self.controller_input_size]

        return ncp_input.to(self.compute_device)

    def process_and_respond(self, user_input_text: str, store_intermediate_states: bool = False):
        # Added: Measure total processing time
        start_time = time.time()
        
        # 1. Prepare input for NCP Controller
        ncp_input_features = self.get_combined_input_for_ncp(user_input_text)
        
        # 2. NCP Controller generates control signals
        ncp_start_time = time.time()
        self.ncp_controller.reset_hidden_states(batch_size=1, device=self.compute_device) # Reset for each new independent interaction
        control_signals = self.ncp_controller(
            ncp_input_features,
            store_activations=store_intermediate_states, 
            store_control_signals=store_intermediate_states
        ) 
        control_signals_text = ", ".join([f"S{i}:{v:.3f}" for i, v in enumerate(control_signals.squeeze().tolist())])
        ncp_time = time.time() - ncp_start_time

        # 3. Generate LLM prompt
        llm_start_time = time.time()
        formatted_history_for_llm = list(self.conversation_history) 
        templated_prompt = self.llm_interface.generate_qwen_prompt(
            task_description=user_input_text,
            control_signals_text=control_signals_text,
            conversation_history_list=formatted_history_for_llm
        )
        
        # 4. Get LLM response
        llm_final_response, llm_thinking_content = self.llm_interface.generate_response(templated_prompt)
        if llm_thinking_content and "<think></think>" not in llm_thinking_content.replace("\n",""):
             print(f"LLM Thinking: {llm_thinking_content}")
        llm_time = time.time() - llm_start_time

        # 5. Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input_text})
        self.conversation_history.append({"role": "assistant", "content": llm_final_response})

        # 6. Evaluate performance
        eval_start_time = time.time()
        metrics = self.performance_monitor.evaluate(llm_final_response, user_input_text)
        eval_time = time.time() - eval_start_time

        # 7. Update Continuous Learning Module (CfC)
        clm_start_time = time.time()
        interaction_text = user_input_text + " [SEP] " + llm_final_response # Add separator
        interaction_features = self._text_to_features(interaction_text, self.continuous_learning.clm_input_size)
        time_step_tensor = torch.tensor([[1.0]], device=self.compute_device) # Represents one discrete step
        
        self.continuous_learning.reset_hidden_state(batch_size=1, device=self.compute_device) # Reset for each new independent interaction
        clm_output = self.continuous_learning(
            interaction_features, 
            time_step_tensor,
            store_history=store_intermediate_states
        )
        clm_time = time.time() - clm_start_time

        # 8. Learn from interaction
        learn_start_time = time.time()
        loss = self._learn_from_interaction(ncp_input_features, control_signals, llm_final_response, clm_output)
        self.learning_losses.append(loss.item())
        learn_time = time.time() - learn_start_time
        print(f"Learning Loss: {loss.item():.4f} | Relevance: {metrics.get('relevance',0):.2f}")

        # 9. Collect data for potential fine-tuning
        ft_start_time = time.time()
        # Prompt for FT should ideally be the user_input + context that would lead to llm_final_response
        context_str_for_ft = " ".join([h['content'] for h in formatted_history_for_llm[-4:-1:2]]) # last 2 user inputs
        ft_prompt = f"Context: {context_str_for_ft}\nUser: {user_input_text}".strip()
        self.ft_data_collector.add_data_point((ft_prompt, llm_final_response), metrics)
        
        if len(self.ft_data_collector.fine_tuning_data) >= 5 : # Save every 5 new points
            self.ft_data_collector.save_data_for_fine_tuning()
        ft_time = time.time() - ft_start_time
            
        # 10. Record component timing metrics
        end_time = time.time()
        total_time = end_time - start_time
        self.processing_times.append({
            "total": total_time,
            "ncp": ncp_time,
            "llm": llm_time,
            "eval": eval_time,
            "clm": clm_time,
            "learn": learn_time,
            "ft": ft_time
        })
        
        # Detailed timing output if needed
        # print(f"Timing: NCP={ncp_time:.3f}s, LLM={llm_time:.3f}s, Learn={learn_time:.3f}s, Total={total_time:.3f}s")
            
        return llm_final_response

    def _learn_from_interaction(self, ncp_input_feats, ncp_ctrl_signals, 
                               llm_resp_txt, clm_out_feats):
        self.optimizer.zero_grad()
        
        # Loss Term 1: NCP predicting features of the LLM's response
        # This helps NCP learn what kind of responses are generated.
        target_resp_feat_len = self.ncp_output_size 
        target_resp_feats = self._text_to_features(llm_resp_txt, target_resp_feat_len)
        
        # Re-evaluate NCP with current input to ensure gradients for ncp_ctrl_signals used for prediction
        # ncp_ctrl_signals is already the output of ncp_controller(ncp_input_feats)
        loss_ncp_predict = self.mse_loss(ncp_ctrl_signals, target_resp_feats)
        
        # Loss Term 2: NCP L1 regularization for sparsity
        loss_ncp_l1 = self.ncp_controller.get_l1_loss()

        # Loss Term 3: Align NCP control signals with CLM's "understanding" of the interaction
        # This encourages control signals that are coherent with the interaction's outcome.
        loss_cl_align = torch.tensor(0.0, device=self.compute_device)
        if self.ncp_output_size == self.continuous_learning.cfc_hidden_size:
            loss_cl_align = self.mse_loss(ncp_ctrl_signals, clm_out_feats)
        else:
            # If sizes don't match, could use a projection, or ignore this loss term.
            # For simplicity, if they don't match, this term is effectively zero unless a projection is added.
            # Or, project one to the other's dimension. Let's project CLM output to NCP output size for now.
            if not hasattr(self, 'clm_to_ncp_proj'):
                self.clm_to_ncp_proj = nn.Linear(self.continuous_learning.cfc_hidden_size, self.ncp_output_size).to(self.compute_device)
                # Add these new parameters to the optimizer
                self.optimizer.add_param_group({'params': self.clm_to_ncp_proj.parameters()})

            projected_clm_out = self.clm_to_ncp_proj(clm_out_feats)
            loss_cl_align = self.mse_loss(ncp_ctrl_signals, projected_clm_out)

        # Weighting the losses
        total_loss = loss_ncp_predict + 0.1 * loss_ncp_l1 + 0.5 * loss_cl_align 
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.ncp_controller.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.continuous_learning.parameters(), max_norm=1.0)
        if hasattr(self, 'clm_to_ncp_proj'):
             torch.nn.utils.clip_grad_norm_(self.clm_to_ncp_proj.parameters(), max_norm=1.0)

        self.optimizer.step()
        return total_loss

    def reset_all_states(self):
        self.ncp_controller.reset_hidden_states(device=self.compute_device)
        self.continuous_learning.reset_hidden_state(device=self.compute_device)
        self.conversation_history.clear()
        self.performance_monitor.metrics.clear() # Clear metrics from monitor
        print("Hybrid Brain states, history, and metrics have been reset.")
    
    # Added: Enable async processing
    def enable_async_processing(self):
        """Enable asynchronous processing mode."""
        if self.async_generator is None:
            self.async_generator = AsyncResponseGenerator(self)
            self.async_generator.start()
            return True
        elif not self.async_generator.running:
            self.async_generator.start()
            return True
        return False
    
    # Added: Disable async processing
    def disable_async_processing(self):
        """Disable asynchronous processing mode."""
        if self.async_generator and self.async_generator.running:
            self.async_generator.stop()
            return True
        return False
    
    # Added: Queue request for async processing
    def queue_request(self, user_input_text, request_id=None):
        """Queue a request for asynchronous processing."""
        if self.async_generator is None or not self.async_generator.running:
            self.enable_async_processing()
        return self.async_generator.queue_request(user_input_text, request_id)
    
    # Added: Get response from async processing
    def get_async_response(self, block=True, timeout=None):
        """Get a response from the asynchronous processing queue."""
        if self.async_generator is None:
            return None, None, None, "Async processing not enabled"
        return self.async_generator.get_response(block, timeout)
    
    # Added: Analyze performance
    def analyze_performance(self, output_file=None):
        """Analyze and visualize performance metrics of the hybrid brain."""
        if not self.processing_times:
            return "No performance data available yet."
            
        # Extract timing data for each component
        components = ['total', 'ncp', 'llm', 'eval', 'clm', 'learn', 'ft']
        component_times = {comp: [t.get(comp, 0) for t in self.processing_times] for comp in components}
        
        # Calculate statistics
        stats = {comp: {
            'avg': np.mean(times),
            'max': np.max(times),
            'min': np.min(times),
            'std': np.std(times),
            'total': np.sum(times),
            'percent': 100 * np.sum(times) / np.sum(component_times['total']) if comp != 'total' else 100
        } for comp, times in component_times.items()}
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Component timing over interactions
        plt.subplot(2, 1, 1)
        for comp in ['ncp', 'llm', 'clm', 'learn', 'eval', 'ft']:  # Skip total for clarity
            plt.plot(component_times[comp], label=f'{comp.upper()} ({stats[comp]["avg"]:.3f}s avg)')
        plt.title('Component Processing Times per Interaction')
        plt.xlabel('Interaction Index')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Average time breakdown as pie chart
        plt.subplot(2, 1, 2)
        labels = [f'{comp.upper()} ({stats[comp]["percent"]:.1f}%)' for comp in ['ncp', 'llm', 'clm', 'learn', 'eval', 'ft']]
        sizes = [stats[comp]['total'] for comp in ['ncp', 'llm', 'clm', 'learn', 'eval', 'ft']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Average Processing Time Breakdown')
        plt.axis('equal')
        
        plt.tight_layout()
        
        # Save to file if requested
        if output_file:
            plt.savefig(output_file)
            
        return stats, plt.gcf()
    
    # Added: Plot learning progress
    def plot_learning_progress(self, window_size=10, output_file=None):
        """Plot the learning progress of the hybrid brain."""
        if not self.learning_losses:
            return "No learning data available yet."
            
        plt.figure(figsize=(10, 6))
        
        # Plot raw losses
        plt.plot(self.learning_losses, label='Raw Loss', alpha=0.3)
        
        # Plot moving average if we have enough data points
        if len(self.learning_losses) >= window_size:
            moving_avg = np.convolve(self.learning_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(moving_avg)), moving_avg, 'r-', label=f'{window_size}-point MA')
            
        plt.title('Learning Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_file:
            plt.savefig(output_file)
            
        return plt.gcf()
    
    # Added: Save brain state
    def save_brain_state(self, file_path="hybrid_brain_state.pt"):
        """Save the state of the hybrid brain to a file."""
        state_dict = {
            'ncp_controller': self.ncp_controller.state_dict(),
            'continuous_learning': self.continuous_learning.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_losses': self.learning_losses,
            'processing_times': self.processing_times
        }
        
        if hasattr(self, 'clm_to_ncp_proj'):
            state_dict['clm_to_ncp_proj'] = self.clm_to_ncp_proj.state_dict()
            
        torch.save(state_dict, file_path)
        print(f"Brain state saved to {file_path}")
        return file_path
    
    # Added: Load brain state
    def load_brain_state(self, file_path="hybrid_brain_state.pt"):
        """Load the state of the hybrid brain from a file."""
        if not os.path.exists(file_path):
            print(f"Error: State file {file_path} not found.")
            return False
            
        try:
            state_dict = torch.load(file_path, map_location=self.compute_device)
            
            self.ncp_controller.load_state_dict(state_dict['ncp_controller'])
            self.continuous_learning.load_state_dict(state_dict['continuous_learning'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            
            if 'learning_losses' in state_dict:
                self.learning_losses = state_dict['learning_losses']
            if 'processing_times' in state_dict:
                self.processing_times = state_dict['processing_times']
                
            if 'clm_to_ncp_proj' in state_dict:
                if not hasattr(self, 'clm_to_ncp_proj'):
                    self.clm_to_ncp_proj = nn.Linear(
                        self.continuous_learning.cfc_hidden_size, 
                        self.ncp_output_size
                    ).to(self.compute_device)
                    self.optimizer.add_param_group({'params': self.clm_to_ncp_proj.parameters()})
                self.clm_to_ncp_proj.load_state_dict(state_dict['clm_to_ncp_proj'])
                
            print(f"Brain state loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading brain state: {e}")
            import traceback
            traceback.print_exc()
            return False

# --- Main Interaction Loop ---
def main():
    print("Initializing Hybrid Brain with Qwen3...")
    
    # CONFIGURATION
    # User's specific local path for the Qwen3 model
    qwen_local_path = r"C:\Users\ali_z\ANU AI\V1\Qwen3-main\Qwen3-0.6B\Qwen3-0.6B"
    
    # Fallback Hugging Face Hub identifier if the local path is not valid
    qwen_hub_identifier = "Qwen/Qwen3-0.6B" 

    controller_feat_size = 192 
    ncp_hid_sizes = [64, 64, 64] 
    ncp_out_size = 32          
    
    clm_feat_size = 128        
    cfc_hid_size = 32 # Matches ncp_out_size for direct alignment loss term

    if ncp_out_size != cfc_hid_size:
        print(f"Warning: NCP output size ({ncp_out_size}) and CfC hidden size ({cfc_hid_size}) are different. "
              "The alignment loss term will use a projection or may need adjustment.")

    brain = HybridBrainQwen(
        controller_input_size=controller_feat_size,
        ncp_hidden_sizes=ncp_hid_sizes,
        ncp_output_size=ncp_out_size,
        clm_input_size=clm_feat_size,
        cfc_hidden_size=cfc_hid_size,
        qwen_model_path_or_name=qwen_hub_identifier, 
        qwen_local_model_dir=qwen_local_path,      
        lr=0.0001 # Further reduced LR
    )

    print("\nANU AI - Hybrid Brain (Qwen3 + NCP/CfC)")
    print("Type 'exit' to quit, 'reset' to reset brain state.")
    print("Additional commands: 'async' for async mode, 'stats' for performance stats, 'save' to save brain state")

    # Added: Command processing
    async_mode = False
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        
        # Command processing
        if user_input.lower() == 'exit':
            print("Exiting...")
            if brain.ft_data_collector.fine_tuning_data:
                 brain.ft_data_collector.save_data_for_fine_tuning()
            
            # Gracefully shutdown async processing if active
            if async_mode:
                brain.disable_async_processing()
                
            break
            
        elif user_input.lower() == 'reset':
            brain.reset_all_states()
            continue
            
        elif user_input.lower() == 'async':
            async_mode = not async_mode
            if async_mode:
                brain.enable_async_processing()
                print("Async processing mode enabled.")
            else:
                brain.disable_async_processing()
                print("Async processing mode disabled.")
            continue
            
        elif user_input.lower() == 'stats':
            print("\nGenerating performance statistics...")
            stats, fig = brain.analyze_performance()
            plt.show()
            continue
            
        elif user_input.lower() == 'save':
            save_path = brain.save_brain_state()
            print(f"Brain state saved to {save_path}")
            continue
            
        elif user_input.lower().startswith('load '):
            file_path = user_input[5:].strip()
            if brain.load_brain_state(file_path):
                print(f"Successfully loaded brain state from {file_path}")
            continue
        
        # Regular processing
        try:
            print("Hybrid Brain Processing...")
            
            if async_mode:
                # Queue request and continue immediately
                request_id = brain.queue_request(user_input)
                print(f"Request queued (ID: {request_id}). Type 'get {request_id}' to retrieve the response when ready.")
            else:
                # Synchronous processing
                assistant_response = brain.process_and_respond(user_input)
                print(f"\nAssistant: {assistant_response}")
                
        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
        
        # Check for async response retrieval command
        if user_input.lower().startswith('get '):
            request_id = user_input[4:].strip()
            print(f"Checking for response to request {request_id}...")
            
            # Wait with timeout for response
            rid, inp, response, error = brain.get_async_response(block=True, timeout=0.5)
            
            if rid == request_id:
                if error:
                    print(f"\nError processing request: {error}")
                else:
                    print(f"\nAssistant response for '{inp}':\n{response}")
            else:
                print("Response not ready yet or request ID not found.")

if __name__ == "__main__":
    main()
