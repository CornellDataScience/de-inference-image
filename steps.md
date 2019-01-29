minikube start

minikube dashboard

eval $(minikube docker-env)

docker image build -t de-inference .

eval $(minikube docker-env -u)

sudo kubectl create -f deployment.yaml

sudo kubectl delete deployment,service --all

sudo kubectl get pods,deployments,services,secrets -l app=de-inference
