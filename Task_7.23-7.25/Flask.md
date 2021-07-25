## Flask

### 入门

#### 路由

​	就是url里面类似于文件路径的东西，把/当成文件夹，无/当成文件（重定向）

```
@app.route('/hello')
```

这样可以通过127.0.0.1:5000/hello 来访问Flask给我们return的数据（如果是get请求的话）

#### 模板

​	创建一个视图函数，将模板的内容返回，（与url_for的差别是，路由不会改变）可以传入模板变量给模板，然后通过render_template返回。

### Flask-SQLAlchemy

​	这块暂时很晕，用的时候老是报错。

### Flask-WTF

​	一个渲染表单的工具，搭配Bootstrap 来使用，创建表单方便。