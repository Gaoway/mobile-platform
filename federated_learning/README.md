在 configs 文件夹下可以添加或者修改配置
在脚本文件 run.sh 里编写要运行的实验配置
执行命令 sh run.sh 即可按序运行各个配置

说明：
1. 未在 json 文件中声明的参数将会被默认配置为 config.py 中 CommonConfig 里的初始化值
2. 在自己的文件夹下运行该代码时，需要将 utils.py 与 datasets.py 中的 jmyan 替换成自己的用户名
3. 建议在基于该代码实现其他功能时，创建一个副本，在副本中进行开发，副本的文件夹名称需要定义为方法的名字
4. 程序的输出在 /data/用户名/record/方法名称 文件夹中，日志的命名规则为 数据集-配置名称_at_运行时间.log