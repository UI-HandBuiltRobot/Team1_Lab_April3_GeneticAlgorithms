# APR3 Lab: ROS2 + Genetic Algorithm xArm Workflow

This repo has two parts:

- `apr3_class_ros2_sim/`: ROS 2 package for URDF + RViz launch
- `apr3_class_python/`: GA solver + optional RViz trajectory publisher + optional hardware execution

---

## 1) Prerequisites

Assumes Ubuntu/Linux with ROS 2 Jazzy installed (matching `.vscode/settings.json`).

If ROS 2 is not sourced automatically in your shell:

```bash
source /opt/ros/jazzy/setup.bash
```

---

## 2) Build ROS environment

Build only this package:

```bash
cd Lab_April3_GeneticAlgorithms
colcon build --symlink-install --packages-select apr3_class_ros2_sim
```


## 3) Create Python virtual environment (venv)

Use `--system-site-packages` so ROS Python modules (like `rclpy`) from apt are visible inside the venv.

In a new terminal:

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib xarm xacro
```

---
## 4) Run the RVIZ

### Launch RViz model view

```bash
source install/setup.bash
ros2 launch apr3_class_ros2_sim disp_GA_marker.launch.py
```

THE SYSTEM WILL LOOK BROKEN. That is okay. It will be fixed when the GA runs.

## 5) Run the GA main file

Main entrypoint: `apr3_class_python/src/ga_main.py`

Recommended run command:

```bash
source venv/bin/activate
source install/setup.bash
cd src/apr3_class_python
python ga_main.py
```

Or just select the run button in VSCode while ga_main.py is open.

### Important runtime toggles (edit in `ga_main.py`)

At the top of the file:

- `GOAL_X`, `GOAL_Y`, `GOAL_Z`: target point in mm
- `VIZ`: when `True`, publishes trajectories to ROS2 `/joint_states`
- `EXECUTE_HARDWARE`: when `True`, commands real xArm through `controller_class.py`
- `OPERATE_HAND`: open/close gripper behavior
- `PLAYBACK_HZ`, `INTERP_STEPS`, `VIZ_SKIP_GENS`: RViz playback behavior

Safe default for simulation-only:

- `VIZ = True`
- `EXECUTE_HARDWARE = False`

---

## 5) Where to put your fitness function

Fitness logic lives in:

- `apr3_class_python/genetic_algorithm.py`
- Class: `GeneAlgo`
- Method: `_compute_reward(self, min_distance, best_state, chromosome_length)`

Inside `_compute_reward`, edit these three values:

- `distance_reward`
- `pose_reward`
- `length_penalty`

Then final score is already combined as:

```python
fitness = distance_reward + pose_reward - length_penalty
```

Feel free to add more rewards and penalties as desired.

You can also tune global GA constants near the top of `genetic_algorithm.py`, e.g.:

- `POPULATION_SIZE`
- `MUTATION_RATE`
- `STEP_SIZE`
- `TARGET_DISTANCE_THRESHOLD_MM`

---

## 6) Typical workflow

1. Start RViz launch (`ros2 launch apr3_class_ros2_sim disp_GA_marker.launch.py`).
2. Run `python ga_main.py`.
3. Check terminal fitness output + `ga_fitness_history.png`.
4. Iterate reward weights and GA constants.

---

## 7) Troubleshooting

- `ImportError: rclpy` inside venv:
  - Recreate venv with `--system-site-packages` and source `/opt/ros/jazzy/setup.bash` first.
- Launch package-not-found for `xarm_viz`:
  - Update launch file package name to `apr3_class_ros2_sim`.
- Hardware control import/USB issues:
  - Keep `EXECUTE_HARDWARE = False` until `xarm` package + USB permissions are confirmed.
