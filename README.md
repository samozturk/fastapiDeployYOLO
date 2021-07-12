
## Running the code on your local machine
If you want to run it locally, just use prediction.py file. Open the terminal or cmd, type:
> python prediction.py

Other option could be containerize the code, deploy it on kubernetes. As I mentioned on my blog post here: [Deploying a ML Model with FastAPI on Google Kubernetes Engine(Google Cloud Platform)](https://medium.com/analytics-vidhya/deploying-a-ml-model-with-fastapi-on-google-kubernetes-engine-google-cloud-platform-bc2adbe0a35a)

In this case, you need to delete last chunk of code which is this:
https://github.com/samozturk/fastapiDeployYOLO/blob/34d210187437a28a2d8f30c7b2a5ce0bc9d901c1/prediction.py#L115-L116
