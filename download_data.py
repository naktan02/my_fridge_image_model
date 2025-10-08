from roboflow import Roboflow

rf = Roboflow(api_key="Sr8Uehvni7HFld4OoYmu")
project = rf.workspace("myfridge-2ey6e").project("my_fridge-4s8uk")
version = project.version(1)

# ✅ 최신 roboflow>=1.1.0 기준
dataset = version.download("yolov11")
print("✅ Dataset path:", dataset.location)
