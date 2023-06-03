import numpy as np
import json
import triton_python_backend_utils as pb_utils
import logging

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        output_config = pb_utils.get_output_config_by_name(self.model_config, "pre_out")

        self.pre_out_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        """
        Convert int32 numpy array to float32
        """

        responses = []

        for request in requests:
            input_data = pb_utils.get_input_tensor_by_name(request, "pre_in")

            input = input_data.as_numpy()

            output = (input / 255.0).astype(np.float32)

            output_tensor = pb_utils.Tensor("pre_out", output.astype(self.pre_out_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        logging.info('Cleaning up...')