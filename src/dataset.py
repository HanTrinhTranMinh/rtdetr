# Xử lý COCO data
# src/dataset.py
# dataset.py

import os
from roboflow import Roboflow


def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version_number: int,
    format_type: str = "coco"
):
    """
    Download dataset from Roboflow.

    Args:
        api_key (str): Roboflow API key
        workspace (str): Workspace name
        project_name (str): Project name
        version_number (int): Version number
        format_type (str): Dataset format (e.g., 'coco', 'yolov5', 'yolov8')

    Returns:
        dataset_path (str): Local path to downloaded dataset
    """

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    print(f"Accessing workspace: {workspace}")
    project = rf.workspace(workspace).project(project_name)

    print(f"Downloading version {version_number} in format: {format_type}")
    version = project.version(version_number)
    dataset = version.download(format_type)

    print("Download complete!")
    print(f"Dataset saved at: {dataset.location}")

    return dataset.location


if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY_HERE"

    dataset_path = download_dataset(
        api_key=API_KEY,
        workspace="gn-nhn-yc6af",
        project_name="printed-circuit-board-olvhh",
        version_number=1,
        format_type="coco"  # đổi sang yolov8 nếu cần
    )

    print("Dataset path:", dataset_path)