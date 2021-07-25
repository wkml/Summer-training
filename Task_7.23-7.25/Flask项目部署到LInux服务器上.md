## Flask项目部署到Linux服务器上

### 准备工作

​	阿里云ECS服务器一台，Xshell（或阿里云实例远程连接），Flask项目（推送到github上）

### Xshell配置

​	Xshell连接到Linux服务器，update包安装环境，验证python版本无问题后，安装Nginx。

​	服务器上添加安全组，将22端口、5000端口添加到安全组中。

​	安装git、安装虚拟环境，进入到虚拟环境中。

​	在虚拟环境中pip install flask；

​	配置ssh秘钥，添加到github上。

​	创建一个空目录，创建git仓库，连接到github仓库，从仓库上pull项目下来。

​	在文件夹上，python xxx.py，运行项目，配置完成。

​	最后一步，pip install uwsgi

​	配置好服务器的ini文件，然后 uwsgi -init ...启动服务器

​	完成！

