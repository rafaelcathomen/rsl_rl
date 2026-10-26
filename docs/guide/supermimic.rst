Integrating SuperMimic with a simulator
========================================

SuperMimic learns a differentiable dynamics model from real physical transitions
and improves a deterministic policy through imagined rollouts. Use it when the
task provides differentiable observations, goals, rewards and termination heads.
The simulator itself does not need to support differentiation.

Implement the physical state representation
--------------------------------------------

Implement :class:`~rsl_rl.env.model_based_env.StateRepresentation` in the
environment package. States are tuples of tensors with matching leading batch
dimensions. The representation defines invariant input features, target
increments, integration and a smooth prediction error. It also decodes replay
tensors and supplies per-component error floors.

For floating-base robots, use frame-local translation and rotation increments
and integrate rotations on SO(3). Keep forward kinematics, joint ordering and
rotation conventions in the environment package. A Euclidean system can instead
use a single-element state tuple and ordinary addition and subtraction.

Provide differentiable task heads
----------------------------------

Implement the interface described by
:class:`~rsl_rl.env.model_based_env.DifferentiableTask`. Observation history and
policy goals must contain only information available to the deployed policy.
Physical parameters belong in normalized world-model context and privileged
critic features.

Evaluate rewards on the successor state using the same task function in real
simulation and imagination. Explicitly advance exogenous goal state in both
paths. Keep gradients through state-dependent goals, forward kinematics and
surrogate rewards. Freezing model weights during actor optimization preserves
derivatives with respect to states and actions.

The optional ``model_loss`` hook supplies an auxiliary prediction loss and
detached validation metrics. Its loss weight defaults to zero. Include physical
parameter names, ordering, normalization ranges and other semantic conventions
in ``checkpoint_metadata()``. Curriculum state belongs in
``training_state_dict()`` and ``load_training_state_dict()``.

Expose physical transitions
----------------------------

Implement :class:`~rsl_rl.env.model_based_env.ModelBasedEnv`. A standard
:class:`~rsl_rl.env.vec_env.VecEnv` wrapper alone is insufficient: the learner
needs physical states, previous actions, physical context, goal state and episode
boundaries. Preserve the true successor before autoreset. Replay copies these
transitions before the simulator can reuse its buffers.

Isaac Lab adapters can obtain physical state from the articulation and scene,
step either a manager-based environment or a dedicated simulation driver, and
provide differentiable task heads alongside it. Keep this adapter in the task
package so RSL-RL has no dependency on Isaac Lab, Newton or a particular robot.

Start the native runner
------------------------

Pass an environment with a ``task`` attribute to
:class:`~rsl_rl.runners.supermimic_runner.SuperMimicRunner`:

.. code-block:: python

   from rsl_rl.runners import SuperMimicRunner

   runner = SuperMimicRunner(
       env,
       {"algorithm": {"horizon": 32}, "save_interval": 2000},
       log_dir="logs/supermimic/run",
       device=str(env.device),
   )
   runner.learn(num_learning_iterations=10000)

The runner uses the existing RSL-RL logger backends. ``SuperMimicCfg`` contains
the learner settings. Defaults fit the model to eight-step replay windows and
optimize a 32-step imagined return. The first collection warms replay; 500
model-only iterations precede actor updates. Network sizes are configurable.
CUDA training uses fused Adam and optional ``torch.compile``. Replay allocation
is bounded by half of free GPU memory by default. CPU execution is supported.

The runner calls ``task.begin_iteration()`` once before collection and imagination.
Custom training loops calling ``learner.iterate()`` must call this hook themselves.

Only single-process training is implemented. Simulator and learner must share a
device. No multi-GPU synchronization or performance equivalence to PPO/SAC is
implied by sharing the library.

Save and export a policy
-------------------------

``runner.save(path)`` records model and optimizer states, iteration, task
metadata and task training state. Full loads require matching task and context
semantics. Replay and simulator state are omitted, so a training load starts
fresh collection rather than reproducing the exact next transition.

``runner.load(path, load_optimizer=False)`` restores all models, task training state
and iteration while retaining the newly constructed optimizers and configured
learning rate. This is separate from policy-only loading.

``runner.load(path, policy_only=True)`` checks task identity, goal dimensions
and the ordered action-to-joint mapping, but permits different privileged
physical context. Both load modes reject a changed action mapping even when
decoder dimensions match. Only load trusted training
checkpoints. ``runner.export_policy_to_jit(directory)`` exports a deterministic
policy, including its input normalizers. The exported inputs are
``observation_history[batch, history, obs_dim]`` and ``goal[batch, goal_dim]``.

The AGILE integration keeps its existing training/evaluation scripts and frozen
policy format through a compatibility adapter. Motion-library evaluation stays
in AGILE. The core algorithm has no motion-library or WBC reward dependency.
