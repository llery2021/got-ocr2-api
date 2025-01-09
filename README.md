# got-ocr2-api
a simple api project use got-ocr2 model  

got-ocr2[官方项目](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)由阶跃星辰多模态团队于2024年9月开源推出，是目前在通用性和识别准确率上表现非常优秀、极具创新的端到端VLMs OCR基础项目。  


本项目是使用llama_cpp推理引擎对got模型进行推理的http API服务项目，不依赖也不嵌入原官方项目，为独立部署运行的第三方演示性项目。

# 安装部署

## 下载安装
````md
# 下载项目，进入项目所在目录（/mnt/d/test为示意目录）
git https://github.com/llery2021/got-ocr2-api.git
git lfs pull # 拉取项目中大文件（权重文件），需要安装git LFS插件
cd /mnt/d/test/got-ocr2-api 
# 使用conda命令新建python虚拟环境
# 最低要求python版本>=3.8，推荐版本3.10.14
conda create --name got_ocr2_api python=3.10.14  # got_ocr2_api为示例名称
# 激活新建的虚拟环境
conda activate got_ocr2_api
# 安装依赖 
pip install torch torchvision # 安装torch
pip install ninja # 安装flash_attn前需要装的
pip install flash_attn --no-build-isolation # 这货不容易安装，出现问题网上搜答案
# 最后安装项目中配置文件指定的其他依赖库
pip install -e .
````

# 下载GOT模型权重文件
模型权重(HF)文件请从 [百度网盘](https://pan.baidu.com/s/1G4aArpCOt6I_trHv_1SE2g) 下载<提取码：OCR2>  
假设模型权重文件下载后存放于本地的路径为 /mnt/d/GOT-OCR2.0/GOT_weights  
修改 **config.json** 文件中两处参数  
（1）architectures 参数从原先的 *GOTQwenForCausalLM* --> *Qwen2GotForCausalLM*  
（2）model_type 参数从原先的 *GOT* --> *qwen2* 修改后示意如下：

````json
{
  "_name_or_path": "none",
  "architectures": [
    "Qwen2GotForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  ...省略其他参数...
  "model_type": "qwen2",
  ...省略其他参数...
}
  ````

## 模型转换
项目中convert_hf_to_gguf.py为模型转换脚本，执行命令示例：
````md
python convert_hf_to_gguf.py --model /mnt/d/GOT-OCR2.0/GOT_weights --outfile /mnt/d/got_model_f16.gguf --outtype bf16
````

## 启动api服务
项目中api_server.py为api服务启动脚本，执行命令示例：
````md
python -m api_server --model-path /mnt/d/got_model_f16.gguf
````


# 接口说明

## 接口方法/地址

````md
http://127.0.0.1:8000/v1/ocr
# 请根据实际部署修改ip、端口或使用域名访问
````
## 请求类型
````md
POST

````

## 请求体json参数
````json
{
    "image_file": "/mnt/d/ocr-test/1.png",
    "type": "plain",
    "max_tokens": 2048,
    "temperature": 0,
    "top_p": 0.9,
    "repetition_penalty": 1
}

````


json请求参数说明： 
````md
image_file 图片文件路径或url
# 参数类型：字符串或字符串数组，支持png、jpg、jpeg、webp格式
# 本地图片请填写图片的本地绝对路径
# 远程图片请填写图片的完整url
# 若使用base64编码（utf-8），请带上完整的data头信息，如下所示：
# data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...
````
____________________________
````md
type 识别模式，取值plain或format
# 取值plain时为纯文本模式，识别结果不含格式化信息
# 取值format时为格式化文本模式，比如数学公式会返回mathpix markdown格式
````
____________________________
````md
max_tokens 识别结果最大tokens数量
# 如果能提前确定图片内最多能识别出多少字，填入紧凑的数值，可提高识别速度
# 如果不确定图片内文字数量，一般设2048足够了
````
____________________________
````md
temperature 温度参数
# 取值范围为0到1，数值越小识别结果越稳定，数值越大识别结果越随机
# 一般情况下请设为 0 以保证同一图片多次输出结果均相同
````
____________________________
````md
top_p 采样参数
# 一般情况填0.9即可（选填参数）
repetition_penalty 模型重复惩罚参数
# 一般情况填1即可（选填参数）
````

## 接口响应
接口返回数据（json格式），示例如下：

```json
{
    "id": "vllm-8084fdf60ce441c1a72a0f67aa5b2cc5",
    "object": "text_completion",
    "created": 1729253015,
    "model": "/mnt/d/GOT-OCR2.0/GOT_weights",
    "choices": [
        {
            "index": 0,
            "text": "这里是接口识别的文本内容结果",
            "logprobs": null,
            "finish_reason": "stop",
            "stop_reason": 151645,
            "prompt_logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 286,
        "total_tokens": 300,
        "completion_tokens": 14
    }
}


```

## 客户端调用示例
1、java调用示例（使用OKHttp）
```java
OkHttpClient client = new OkHttpClient().newBuilder().build();
MediaType mediaType = MediaType.parse("application/json");
RequestBody body = RequestBody.create(mediaType, "{\r\n    \"image_file\": \"/mnt/d/ocr-test/1.png\",\r\n    \"type\": \"format\",\r\n    \"max_tokens\": 2048,\r\n    \"temperature\": 0,\r\n    \"top_p\": 0.9,\r\n    \"repetition_penalty\": 1\r\n}");
Request request = new Request.Builder()
   .url("http://127.0.0.1:8000/v1/ocr")
   .method("POST", body)
   .addHeader("Content-Type", "application/json")
   .addHeader("Accept", "*/*")
   .addHeader("Connection", "keep-alive")
   .build();
Response response = client.newCall(request).execute();

````
2、c#调用示例（使用RestSharp）

````csharp
var client = new RestClient("http://127.0.0.1:8000/v1/ocr");
client.Timeout = -1;
var request = new RestRequest(Method.POST);
request.AddHeader("Content-Type", "application/json");
request.AddHeader("Accept", "*/*");
request.AddHeader("Connection", "keep-alive");
var body = @"{
" + "\n" +
@"    ""image_file"": ""/mnt/d/ocr-test/1.png"",
" + "\n" +
@"    ""type"": ""format"",
" + "\n" +
@"    ""max_tokens"": 2048,
" + "\n" +
@"    ""temperature"": 0,
" + "\n" +
@"    ""top_p"": 0.9,
" + "\n" +
@"    ""repetition_penalty"": 1
" + "\n" +
@"}";
request.AddParameter("application/json", body,  ParameterType.RequestBody);
IRestResponse response = client.Execute(request);
Console.WriteLine(response.Content);
````
3、python调用示例（使用requests）

````python
import requests
import json
url = "http://127.0.0.1:8000/v1/ocr"
payload = json.dumps({
   "image_file": "/mnt/d/ocr-test/1.png",
   "type": "format",
   "max_tokens": 2048,
   "temperature": 0,
   "top_p": 0.9,
   "repetition_penalty": 1
})
headers = {
   'Content-Type': 'application/json',
   'Accept': '*/*',
   'Connection': 'keep-alive'
}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)
````
4、php调用示例（使用Guzzle）

````php
<?php
$client = new Client();
$headers = [
   'Content-Type' => 'application/json',
   'Accept' => '*/*',
   'Connection' => 'keep-alive'
];
$body = '{
   "image_file": "/mnt/d/ocr-test/1.png",
   "type": "format",
   "max_tokens": 2048,
   "temperature": 0,
   "top_p": 0.9,
   "repetition_penalty": 1
}';
$request = new Request('POST', 'http://127.0.0.1:8000/v1/ocr', $headers, $body);
$res = $client->sendAsync($request)->wait();
echo $res->getBody();
````

5、javascript调用示例（使用Axios）

````javascript
var axios = require('axios');
var data = JSON.stringify({
   "image_file": "/mnt/d/ocr-test/1.png",
   "type": "format",
   "max_tokens": 2048,
   "temperature": 0,
   "top_p": 0.9,
   "repetition_penalty": 1
});
var config = {
   method: 'post',
   url: 'http://127.0.0.1:8000/v1/ocr',
   headers: { 
      'Content-Type': 'application/json', 
      'Accept': '*/*', 
      'Connection': 'keep-alive'
   },
   data : data
};
axios(config)
.then(function (response) {
   console.log(JSON.stringify(response.data));
})
.catch(function (error) {
   console.log(error);
});

````

<b>:point_right:特别提醒：由于llama_cpp推理got模型因模型转换后识别精度有损失，本项目仅限于演示交流用。</b>