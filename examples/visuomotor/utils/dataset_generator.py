import numpy
from pathlib import Path

def create_folder_structure(dataset_path: str, episode: int) -> dict:
    episode_path = Path(dataset_path).joinpath("episode_" + str(episode))
    create_folder(episode_path)

    # --------------------------------------------------
    joint_position_path = episode_path.joinpath("joint_position")
    create_folder(joint_position_path)

    robot_pose_path = episode_path.joinpath("robot_pose")
    create_folder(robot_pose_path)

    robot_image_path = episode_path.joinpath("image")
    create_folder(robot_image_path)

    # --------------------------------------------------
    left_image_path = robot_image_path.joinpath("left")
    create_folder(left_image_path)

    right_image_path = robot_image_path.joinpath("right")
    create_folder(right_image_path)

    depth_image_path = robot_image_path.joinpath("depth")
    create_folder(depth_image_path)

    print("Saved data for < Episode : ", str(episode), " >")
    path_dict = {"episode_path": episode_path,
                 "joint_position_path": joint_position_path,
                 "robot_pose_path": robot_pose_path,
                 "robot_image_path": robot_image_path,
                 "left_image_path": left_image_path,
                 "right_image_path": right_image_path,
                 "depth_image_path": depth_image_path}

    return path_dict


def create_folder(path: str):
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir()
        return True

    return False


if __name__ == "__main__":
    create_folder_structure("/home/anirudh/Desktop", 1)
