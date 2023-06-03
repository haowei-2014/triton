import numpy as np
import json
import triton_python_backend_utils as pb_utils
import logging

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        output_config = pb_utils.get_output_config_by_name(self.model_config, "post_out")

        self.post_out_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        """
        Get the class id with the highest probability
        """

        responses = []

        for request in requests:
            input_data = pb_utils.get_input_tensor_by_name(request, "post_in")

            input = input_data.as_numpy()

            output = np.argmax(input, axis=1)

            output_tensor = pb_utils.Tensor("post_out", output.astype(self.post_out_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        logging.info('Cleaning up...')