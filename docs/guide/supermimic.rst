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

Keep robot geometry, forward kinematics and rotation conventions in the task
package. The learned dynamics network and its optimization belong to RSL-RL.

Provide differentiable task heads
----------------------------------

Implement the interface described by
:class:`~rsl_rl.env.model_based_env.DifferentiableTask`. Observation history and
policy goals must contain only information available to the deployed policy.
Physical parameters belong in normalized world-model context and privileged
critic features.

Use the same successor reward and goal-advancement semantics in simulation and
imagination. Task methods must preserve derivatives through state-dependent
goals, forward kinematics and surrogate rewards.

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

Keep simulator state capture in the environment adapter. RSL-RL does not depend
on Isaac Lab, Newton or a particular robot.

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

The runner uses the existing RSL-RL logger backends and constructs the algorithm
through :meth:`~rsl_rl.algorithms.supermimic.SuperMimic.construct_algorithm`.
:class:`~rsl_rl.algorithms.supermimic.SuperMimicCfg` defines training and network
settings. Custom training code can pass existing ``world_model``, ``policy`` and
``critic`` modules to the learner constructor. Their interfaces are documented in
:mod:`rsl_rl.models.supermimic_model`.

The runner calls ``task.begin_iteration()`` once before collection and imagination.
Custom training loops calling ``learner.iterate()`` must call this hook themselves.

Training supports one process with simulator and learner on the same device.

Save and export a policy
-------------------------

``runner.save(path)`` records model and optimizer states, iteration, task
metadata and task training state. Full loads require matching task and context
semantics. Replay and simulator state are omitted, so a training load starts
fresh collection rather than reproducing the exact next transition.
Automatic checkpoints use ``model_<iteration>.pt``.

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
