#!/usr/bin/env python3
"""
xArm 1s Single Pose Recorder (pip install xarm)

Records single 6-servo poses one at a time and prints them immediately 
to the terminal in both Raw Counts and Degrees. 
Allows for continuous recording until canceled.
"""

from __future__ import annotations

import time
from typing import List

import xarm

# ---------- GUI (tkinter) ----------
import tkinter as tk
from tkinter import messagebox


SERVO_IDS = [1, 2, 3, 4, 5, 6]


def ask_ok_cancel(title: str, msg: str) -> bool:
    """Return True if OK, False if Cancel."""
    return messagebox.askokcancel(title, msg)


def ask_yes_no(title: str, msg: str) -> bool:
    """Return True if Yes, False if No."""
    return messagebox.askyesno(title, msg)


def read_all(arm: xarm.Controller) -> List[int]:
    """Read all 6 servo positions as a list in servo-id order."""
    vals = [arm.getPosition(servo_id) for servo_id in SERVO_IDS]
    # Some libs return tuples/lists; ensure plain ints.
    return [int(v) for v in vals]


def counts_to_angles(raw_counts: List[int]) -> List[float]:
    """
    Converts raw servo counts (0-1000) to degrees (-120 to 120).
    Assumes 500 is 0 degrees and 1 count = 0.24 degrees.
    """
    angles = []
    for count in raw_counts:
        # Calculate angle and round to 2 decimal places for clean output
        angle = (count - 500) * 0.24
        angles.append(round(angle, 2))
    return angles


def main() -> int:
    # Create hidden Tk root so message boxes work cleanly
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # --- Connect ---
    try:
        arm = xarm.Controller("USB")
    except Exception as e:
        messagebox.showerror("Connection Error", f"Failed to connect via USB:\n\n{e}")
        return 1

    # --- Disable all servos (go limp) so the arm can be moved by hand ---
    try:
        arm.servoOff()
    except Exception as e:
        messagebox.showwarning(
            "Servo Off Warning",
            f"Could not disable servos with arm.servoOff().\n"
            f"You can still proceed, but the arm may resist being moved.\n\n{e}",
        )

    time.sleep(0.2)
    
    print("\n# ================= SINGLE POSE RECORDER =================")
    pose_count = 1

    # --- Continuous Recording Loop ---
    while True:
        if not ask_ok_cancel(
            f"Pose Recorder - Pose #{pose_count}",
            "Move the arm to the desired position.\n\n"
            "Click OK to record (Cancel to quit)."
        ):
            break # User clicked cancel, exit the loop

        # Record and calculate poses
        current_pose_raw = read_all(arm)
        current_pose_angles = counts_to_angles(current_pose_raw)
        
        pose_name = f"POSE_{pose_count}"
        
        # Print both formats to terminal
        print(f"{pose_name}_RAW    = {current_pose_raw}")
        print(f"{pose_name}_ANGLES = {current_pose_angles}")
        print("-" * 50)
        
        messagebox.showinfo(
            "Recorded", 
            f"Recorded {pose_name}:\nRaw: {current_pose_raw}\nAngles: {current_pose_angles}\n\nPrinted to terminal!"
        )

        # Ask if they want to do another
        if not ask_yes_no("Continue?", "Would you like to record another position?"):
            break

        pose_count += 1

    print("# ========================== DONE ========================\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())