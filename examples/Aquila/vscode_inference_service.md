## 该项目是基于[CodeGeeX](https://github.com/CodeGeeX/codegeex-vscode-extension) master分支-tag：1.1.2完成的开发，具体修改如下：

* packagge.json中contributes.configuration.properties添加插件配置信息
```
"CodeGeeX.DecodingStrategies.max_seq_len": {
    "type": "number",
    "default": 50,
    "maximum": 300,
    "minimum": 20,
    "description": "custom the max_seq_len,rang [20, 300]."
},
"CodeGeeX.DecodingStrategies.url": {
    "type": "string",
    "default": "请求模型接口地址",
    "description": "Replace plug-in interface calls"
},
```

* src/params/configures增加获取配置信息
```
const defaultConfig = {
    temp: 0.8,
    topp: 0.95,
    topk: 0,
    url: '默认配置模型输出接口',
    max_seq_len: 50,  // 新增配置
};
const modelConfig = configuration.get("DecodingStrategies", defaultConfig);
export const temp = modelConfig.temp;
export const topk = modelConfig.topk;
export const topp = modelConfig.topp;
export const API_URL = modelConfig.url;  // 获取配置
export const max_seq_len = modelConfig.max_seq_len;  // 获取配置
```

* src/provider/inlineCompletionProvider中去掉权限密钥校验相关代码
```
// rs = await getCodeCompletions(
//     textBeforeCursor,
//     num,
//     lang,
//     apiKey,
//     apiSecret,
//     "inlinecompletion"
// );
let timestart = new Date().getTime();
let timeend = new Date().getTime();
const completions = Array<string>();
let commandid = "";
rs = { completions, commandid };
```

* src/utils/getCodeCompletions 不同模式下接口地址配置，这里我们修改为调用同一个地址，并且地址在插件中可以配置
```
import {
    temp, topp, topk, API_URL,
    max_seq_len, } from "../param/configures";
去掉相关选择代码
 // let API_URL = "";
// if (mode === "prompt") {
//     API_URL = `${apiHref}/multilingual_code_generate_block`;
// } else if (mode === "interactive") {
//     API_URL = `${apiHref}/multilingual_code_generate_adapt`;
// } else {
//     if (generationPreference === "line by line") {
//         API_URL = `${apiHref}/multilingual_code_generate`;
//     } else {
//         API_URL = `${apiHref}/multilingual_code_generate_adapt`;
//     }
// }
payload参数也进行修改
let payload = {
    ability: "seo_article_creation",
    context: prompt,
    temperature: temp,
    top_k: topk,
    top_p: topp,
    max_seq_len: max_seq_len,
    len_penalty: 1.0,
    repetition_penalty: 1.0,
    presence_penalty: 1.0,
    frequency_penalty: 1.0,
    end_tokens: [],
};
接口返回数据格式根据需要自定义
// 原代码位置
 if (res?.data.status === 0) {
    let codeArray = res?.data.result.output.code;
    const completions = Array<string>();
    for (let i = 0; i < codeArray.length; i++) {
        const completion = codeArray[i];
        let tmpstr = completion;
        if (tmpstr.trim() === "") continue;
        if (completions.includes(completion)) continue;
        completions.push(completion);
    }
    let timeEnd = new Date().getTime();
    console.log(timeEnd - time1, timeEnd - time2);
    resolve({ completions, commandid });
} else {
    try {
        await getEndData(commandid, res.data.message, "No");
    } catch (err) {
        console.log(err);
    }
    reject(res.data.message);
}
替换
let codeArray = res?.data.generated;
const completions = Array<string>();
completions.push(codeArray);
let timeEnd = new Date().getTime();
resolve({ completions, commandid });
```

## 项目运行以及打包
* 首先电脑安装前端项目启动所需环境：node。安装完成之后通过node -v和npm -v，安装成功应该可以看到对应的版本号
* 终端目录指向项目根目录，执行`npm install`安装相关以来,安装完成根目录会多出一个package-lock.json文件和node_modules文件夹
* vscode中运行和调试左上角执行`Run Extentensions`,修改src目录下的代码会进行实时编译到out文件夹中进行本地调试
* 安装全局模块vsce [`npm install --global @vscode/vsce`](https://www.npmjs.com/package/@vscode/vsce)
* 修改完代码执行`vsce package`打包插件，根目录会生成.vsix文件，然后在vscode插件中选择从VSIX安装，选择刚刚打包好的文件进行安装试用