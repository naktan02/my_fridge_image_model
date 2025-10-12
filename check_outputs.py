# check_outputs.py
import sys, tensorflow as tf
m = sys.argv[1] if len(sys.argv) > 1 else "model_with_metadata.tflite"
i = tf.lite.Interpreter(model_path=m)
i.allocate_tensors()
print("INPUT :", i.get_input_details())
print("OUTPUT:", len(i.get_output_details()))
for idx, od in enumerate(i.get_output_details()):
    print(idx, od["name"], od["shape"], od["dtype"])
