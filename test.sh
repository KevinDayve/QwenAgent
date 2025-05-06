# curl -X POST   "https://phronetic-ai--qwen-2vl-dapo-multimodal.modal.run"   -H "Content-Type: application/json"   -d '{ 
#         "prompt": "Explain the theory of relativity in simple terms.", "image": "dummy"
#       }'

#       curl -X POST   "https://phronetic-ai--qwen-2vl-dapo-multimodal.modal.run?prompt=Explain the theory of relativity in simple terms."

    
curl -X POST -H 'Content-Type: application/json' --data-binary '{"prompt": "Describe the meaning of life."}' https://phronetic-ai--qwen-2vl-dapo-multimodal.modal.run