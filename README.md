# 华为atlas自定义算子开发
## 环境配置
```
安装CANN(查看官方文档)
```
## 算子开发

### 算子开发流程
```
1.使用msOpGen工具生成算子框架 
示例:${CANN_DIR}/python/site-packages/bin/msopgen gen -i add_custom.json -c ai_core-ascend310b -lan cpp -out ./AddCustom
2.填充op_host op_kernel文件
3.调用build.sh编译算子
4.执行.run文件部署算子
5.单算子API调用
```
