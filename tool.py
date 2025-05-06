from agno.tools import Toolkit
import json
import requests

class ModelHTTP(Toolkit):
    def __init__(self):
        super().__init__(name="image_describer")
        self.register(self.describeImage)

    def describeImage(self, prompt: str, imageUrl: str) -> str:
        """
        Hits a specified endpoint with a given prompt and an optional image string.

        Parameters:
        - prompt: The text prompt to send to the API.
        - image: An optional image string to include in the request.

        Returns:
        - The response from the API as a string.
        """
        url = "https://phronetic-ai--qwen-2vl-dapo-multimodal.modal.run"
        headers = {'Content-Type': 'application/json'}
        data = {'prompt': prompt}
        
        if imageUrl:
            data['image'] = imageUrl
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return str(e)