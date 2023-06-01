# github action 上的容器自动构建

在 github action 中配置秘钥，就可以实现自动构建推送容器

* DOCKER_ORG
  组织名

* DOCKER_USERNAME
  hub.docker.com 登录的用户

* DOCKER_PASSWORD
  hub.docker.com 登录的密码

详情见[../.github/workflows/onnx.yml](../.github/workflows/onnx.yml)。

修改 [./version.txt](./version.txt) 可以设置推送容器的版本号。
