from gr00t_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gr00t_wbc.control.teleop.solver.hand.g1_inspire_gripper_ik_solver import (
    G1InspireGripperIKSolver,
)


# initialize hand ik solvers for g1 robot
def instantiate_g1_hand_ik_solver(hand_type: str = "dex3"):
    """
    Instantiate hand IK solvers for G1 robot.
    
    Args:
        hand_type: "dex3" for three-finger Dex3 hands, "inspire" for Inspire hands
    
    Returns:
        Tuple of (left_hand_ik_solver, right_hand_ik_solver)
    """
    if hand_type == "inspire":
        left_hand_ik_solver = G1InspireGripperIKSolver(side="left")
        right_hand_ik_solver = G1InspireGripperIKSolver(side="right")
        print(f"Using Inspire hand IK solvers")
    else:  # Default to dex3
        left_hand_ik_solver = G1GripperInverseKinematicsSolver(side="left")
        right_hand_ik_solver = G1GripperInverseKinematicsSolver(side="right")
        print(f"Using Dex3 hand IK solvers")
    
    return left_hand_ik_solver, right_hand_ik_solver
