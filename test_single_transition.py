from pathlib import Path
from ot_wo_two_phase_single_transition import OT_WO_Two_Phase
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang
from ot_backprop_pnwo.evaluation.evaluation_param import ConvergenceConfig
import tensorflow as tf
import numpy as np

# Set paths
path_log = Path('./data/Depart.xes')
path_pn = Path('./data/ori.pnml')

# 1. Initialize data containers
print("Initializing SPN and Log...")
spn_wrapper = SPNWrapper(path_pn)
act_id_conn = ActivityIDConnector.from_spn(spn_wrapper.net)
stoch_lang_log = StochasticLang.create_from_log_file(path_log, act_id_conn)

# 2. Initialize optimizer
print("Initializing Optimizer...")
ot_wo_instance = OT_WO_Two_Phase(
    spn_container=spn_wrapper,
    stoch_lang_log=stoch_lang_log,
    act_id_conn=act_id_conn,
    hot_start=False,
    max_nbr_paths=300,
    max_nbr_variants=300,
    run_phase_two=True,
    nbr_samples_phase_two=10
)

# 3. Get initial weights
initial_weights = spn_wrapper.get_weights()
print("Initial weights:", initial_weights)

# 4. Specify target transition
# Let's pick a transition that is likely to be optimized, or just the first one
target_trans_name = spn_wrapper.net.transitions[0].name 
print(f"Optimizing ONLY transition: {target_trans_name}")

# 5. Run optimization
print("Running Optimization...")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
result = ot_wo_instance.optimize_weights(
    optimizer=optimizer,
    target_transition_name=target_trans_name,
    nbr_iterations_min=50,
    nbr_iterations_max=200 
)

# 6. Verify results
final_weights = result.spn_weights
print("Final weights:", final_weights)

# Find index of target transition
trans_names = [t.name for t in spn_wrapper.net.transitions]
target_idx = trans_names.index(target_trans_name)

# Check which weights changed
# We use a small epsilon because floating point comparisons
changed_indices = np.where(np.abs(initial_weights - final_weights) > 1e-6)[0]

print("\n--- Verification Results ---")
if len(changed_indices) == 1 and changed_indices[0] == target_idx:
    print("SUCCESS: Only the target transition weight was updated!")
elif len(changed_indices) == 0:
    print("WARNING: No weights were updated. This might happen if the gradient was zero or optimization didn't step far enough.")
    print("Try increasing learning rate or iterations if you want to see movement.")
else:
    print(f"FAILURE: Weights updated at indices: {changed_indices}")
    print(f"Expected only index: {target_idx} ({target_trans_name})")
    
    for idx in changed_indices:
        print(f"  Index {idx} ({trans_names[idx]}): {initial_weights[idx]} -> {final_weights[idx]}")
