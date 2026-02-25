from roboflow import Roboflow
rf = Roboflow(api_key="29cYOOTGlTb1nYSVgUVq")
project = rf.workspace("gn-nhn-yc6af").project("printed-circuit-board-olvhh")
version = project.version(1)
dataset = version.download("coco", location="./Dataset")
print(f"Dataset đã được tải tại: {dataset.location}")