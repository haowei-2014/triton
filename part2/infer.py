import sys
import numpy as np
import tritonclient.http as httpclient
from PIL import Image


def infer(url: str, model_name: str, input_data: list,
          input_datatype: str, input_name: str, output_name: str):
    try:
        triton_client = httpclient.InferenceServerClient(url=url)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    inputs = []
    outputs = []

    inputs.append(httpclient.InferInput(input_name, input_data.shape, input_datatype))
    inputs[0].set_data_from_numpy(input_data, binary_data=False)

    outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=False))

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    return results.as_numpy(output_name)


image_file = "images/img_94.jpg"
image = Image.open(image_file)
image = np.array(image).astype(np.int32)
image = np.expand_dims(image, axis=0)


result = infer(url="localhost:8000",
               model_name="ensemble",
               input_data=image,
               input_datatype="INT32",
               input_name="input",
               output_name="output")

print(result[0])
