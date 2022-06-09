

Build docker image
~~~
docker build . -t v20100/epimap
~~~

Install bitnami helm chart repo
~~~
helm repo add bitnami https://charts.bitnami.com/bitnami
~~~

Install nginx chart with epimap sidecar
~~~
helm upgrade --install epimap bitnami/nginx -f k8s/values.yaml
~~~
