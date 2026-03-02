#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
car_brain — Autonomous driving package for BFMC.

Modules
-------
  config          — Central configuration dataclass & YOLO class constants
  perception      — Threaded YOLOv8 engine, image conversion, traffic light HSV
  lane_detection  — BEV perspective transform + sliding-window lane fitting
  fsm_logic       — Finite State Machine (INIT → LANE_FOLLOWING → …)
  control         — PID controller + speed ramp + EMA filter
  perception_node — ROS 2 node: camera → YOLO + lane detection → publish
  control_state_node — ROS 2 node: FSM + PID → /cmd_vel
"""
